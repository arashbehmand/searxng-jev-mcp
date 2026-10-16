"""Stage E: merge and dedupe SearXNG results, plus the site helpers later stages need.

Results from the general, vertical and variant requests are grouped by an identifier key
(arXiv id, DOI, PMID/PMC, Stack Exchange question, YouTube id, GitHub repo) when the URL has one,
else by a canonical URL key. The canonical key ignores the scheme, ``www.``, mobile and AMP
variants, default ports, fragments and denylisted tracking parameters, and never anything that
can carry identity. Also here: registrable-domain and site keys, ``site:`` / ``-term`` post-filters,
top-k diversity with backfill, and opaque doc ids. Stdlib only; nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, quote, unquote, unquote_plus, urlsplit

from searxng_jev_mcp.models import Doc, ParsedQuery, RawResult

# --- URL parsing -----------------------------------------------------------------------------

_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_PCT_RE = re.compile(r"%([0-9A-Fa-f]{2})")
_NEEDS_ESCAPE_RE = re.compile(r"[^\x21-\x7e]")  # spaces, controls and non-ASCII (IRI form)
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_HOST_FOLD_PREFIXES = frozenset({"www", "m", "mobile", "amp"})

# Google AMP cache: https://<x>.cdn.ampproject.org/c/s/<host>/<path> (c=content, v=viewer, ...).
_AMP_CACHE_PATH_RE = re.compile(r"^/[cvir]/(s/)?([^/]+\.[^/]+(?:/.*)?)$")
# Google AMP viewer: https://www.google.com/amp/s/<host>/<path>
_GOOGLE_AMP_PATH_RE = re.compile(r"^/amp/(s/)?([^/]+\.[^/]+(?:/.*)?)$")
_AMP_PATH_RE = re.compile(r"^(.*[^/])/amp/?$")  # /story/amp or /story/amp/; a bare /amp is kept
# Code hosts where a trailing /amp is a repo, file or directory name, never an AMP variant.
_NO_AMP_PATH_HOSTS = frozenset({"github.com", "gitlab.com", "codeberg.org", "bitbucket.org"})
_AMP_HTML_RE = re.compile(r"(?<=[^/])\.amp\.html$")

# Query parameters that never carry page identity. Matched on the lowercased, decoded name.
_TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "gclsrc",
        "dclid",
        "msclkid",
        "igshid",
        "yclid",
        "twclid",
        "ttclid",
        "wbraid",
        "gbraid",
        "srsltid",  # appended by Google to result links
        "_hsenc",
        "_hsmi",
        "_ga",
        "_gl",
        "mkt_tok",
        "ref_src",
        # AMP viewer bookkeeping
        "amp_js_v",
        "amp_gsa",
        "amp_ct",
        "amp_tf",
        "usqp",
        "ampshare",
    }
)
_TRACKING_PREFIXES = ("utm_", "mc_")


@dataclass(frozen=True, slots=True)
class _URL:
    scheme: str
    raw_host: str  # lowercased, trailing dot removed, IPv6 kept in brackets
    host: str  # raw_host with www./m./mobile./amp. folded
    port: str | None
    path: str
    query: str
    fragment: str


def _fold_host(host: str) -> str:
    if host.startswith("[") or _IPV4_RE.match(host):
        return host
    labels = host.split(".")
    while len(labels) > 2 and labels[0] in _HOST_FOLD_PREFIXES:
        labels.pop(0)
    # Wikimedia-style mobile hosts (en.m.wikipedia.org): drop an inner "m" label, never one of
    # the last two labels.
    last_two = len(labels) - 2
    return ".".join(label for i, label in enumerate(labels) if label != "m" or i >= last_two)


def _split(url: str) -> _URL | None:
    parts = urlsplit(url)
    if not parts.netloc:
        return None
    hostport = parts.netloc.rpartition("@")[2]  # userinfo never identifies a page
    if hostport.startswith("["):
        end = hostport.find("]")
        if end < 0:
            return None
        host, rest = hostport[: end + 1], hostport[end + 1 :]
        port = rest[1:] if rest.startswith(":") else ""
    else:
        host, _, port = hostport.partition(":")
    host = host.lower().rstrip(".")
    if not host:
        return None
    return _URL(
        scheme=parts.scheme.lower(),
        raw_host=host,
        host=_fold_host(host),
        port=port or None,
        path=parts.path,
        query=parts.query,
        fragment=parts.fragment,
    )


def _unwrap_amp(u: _URL) -> _URL | None:
    """Return the origin URL behind a Google AMP cache or AMP viewer URL, else None."""
    if u.host.endswith(".cdn.ampproject.org"):
        match = _AMP_CACHE_PATH_RE.match(u.path)
    elif u.host == "google.com":
        match = _GOOGLE_AMP_PATH_RE.match(u.path)
    else:
        return None
    if match is None:
        return None
    origin = ("https://" if match.group(1) else "http://") + match.group(2)
    if u.query:
        origin += "?" + u.query
    if u.fragment:
        origin += "#" + u.fragment
    return _split(origin)


def _parse(url: str) -> _URL | None:
    try:
        u = _split(url.strip())
        if u is None:
            return None
        return _unwrap_amp(u) or u
    except (ValueError, UnicodeError):
        return None


# --- canonical URL ---------------------------------------------------------------------------


def _fix_escape(match: re.Match[str]) -> str:
    char = chr(int(match.group(1), 16))
    return char if char in _UNRESERVED else "%" + match.group(1).upper()


def _normalize_escapes(text: str) -> str:
    text = _NEEDS_ESCAPE_RE.sub(lambda m: quote(m.group(0), safe=""), text)
    return _PCT_RE.sub(_fix_escape, text)


def _is_dropped_param(name: str, value: str) -> bool:
    key = unquote_plus(name).strip().lower()
    if key in _TRACKING_PARAMS or key.startswith(_TRACKING_PREFIXES):
        return True
    val = unquote_plus(value).strip().lower()
    if key == "amp":
        return val in ("", "1", "true")
    return key == "outputtype" and val == "amp"


def _canonical_query(query: str) -> str:
    items: list[tuple[str, str]] = []
    for raw_piece in query.split("&"):
        if not raw_piece:
            continue
        piece = _normalize_escapes(raw_piece)
        name, _, value = piece.partition("=")
        if not _is_dropped_param(name, value):
            items.append((name, piece))
    items.sort()
    return "&".join(piece for _, piece in items)


def _canonical_port(scheme: str, port: str | None) -> str:
    if not port:
        return ""
    if not port.isdigit():
        return port.lower()
    number = int(port)
    if (number == 80 and scheme in ("http", "")) or (number == 443 and scheme in ("https", "")):
        return ""
    return str(number)


def _idna(host: str) -> str:
    if host.isascii():
        return host
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def canonical_url(url: str) -> str:
    """Dedupe key for a URL. Never shown to users and never raises.

    Ignores the scheme, ``www.``, mobile and AMP variants, default ports, fragments (except
    ``#!``/``#/`` hash routes) and denylisted tracking parameters; sorts the remaining
    parameters; lowercases the host only. Unparseable input comes back stripped and lowercased.
    """
    fallback = url.strip().lower()
    try:
        u = _parse(url)
        if u is None:
            return fallback
        path = _normalize_escapes(u.path) or "/"
        path = _AMP_HTML_RE.sub(".html", path)
        amp = None if u.host in _NO_AMP_PATH_HOSTS else _AMP_PATH_RE.match(path)
        if amp:
            path = amp.group(1)
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        port = _canonical_port(u.scheme, u.port)
        key = _idna(u.host) + (f":{port}" if port else "") + path
        query = _canonical_query(u.query)
        if query:
            key += "?" + query
        if u.fragment.startswith(("!", "/")):
            key += "#" + _normalize_escapes(u.fragment)
        if u.scheme not in ("http", "https", ""):
            key = f"{u.scheme}://{key}"
        return key
    except Exception:  # never raise: any surprise falls back to the plain lowercased string
        return fallback


# --- identifier keys -------------------------------------------------------------------------

_ARXIV_HOSTS = frozenset({"arxiv.org", "ar5iv.org"})
_ARXIV_PATH_RE = re.compile(
    r"^/(?:abs|pdf|html|format|ps)/(?:arxiv:)?"
    r"(?P<id>\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[a-z]{2})?/\d{7})"
    r"(?:v\d+)?(?:\.pdf)?/?$",
    re.IGNORECASE,
)

_DOI_HOSTS = frozenset({"doi.org", "dx.doi.org"})
_DOI_SHAPE_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_DOI_ORG_PATH_RE = re.compile(r"^/(10\.\d{4,9}/.+)$")
_PUBLISHER_DOI_RE = re.compile(
    r"/doi/(?:(?:abs|full|pdf|epdf|pdfdirect|fulltext|reader|epub|book)/)?(10\.\d{4,9}/.+)$",
    re.IGNORECASE,
)
_SPRINGER_DOI_RE = re.compile(r"^/(?:article|chapter|content/pdf)/(10\.\d{4,9}/.+?)(?:\.pdf)?$")
_DOI_TRAILING_SEGMENT_RE = re.compile(r"/(?:abstract|full|pdf|epdf|summary)$", re.IGNORECASE)

_PUBMED_PATH_RE = re.compile(r"^/(\d{1,9})/?$")
_NCBI_PUBMED_PATH_RE = re.compile(r"^/pubmed/(\d{1,9})/?$")
_PMC_ARTICLE_PATH_RE = re.compile(r"^(/pmc)?/articles/pmc(\d+)(?:/(?:pdf(?:/.*)?)?)?$", re.IGNORECASE)
_EUROPEPMC_PATH_RE = re.compile(r"^/(?:article|abstract)/(med|pmc)/(?:pmc)?(\d+)/?$", re.IGNORECASE)

_SE_ROOTS = frozenset(
    {
        "stackoverflow.com",
        "stackexchange.com",
        "superuser.com",
        "serverfault.com",
        "askubuntu.com",
        "mathoverflow.net",
        "stackapps.com",
    }
)
_SE_SUBSITE_LABELS = frozenset({"meta", "ru", "pt", "es", "ja"})
_SE_QUESTION_PATH_RE = re.compile(r"^/(?:questions|q)/(\d+)(?:/.*)?$")

_YOUTUBE_HOSTS = frozenset({"youtube.com", "music.youtube.com", "youtube-nocookie.com"})
_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_PATH_RE = re.compile(r"^/(?:shorts|embed|live|v|e)/([A-Za-z0-9_-]{11})/?$")
_YOUTU_BE_PATH_RE = re.compile(r"^/([A-Za-z0-9_-]{11})/?$")

_GITHUB_OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_GITHUB_RESERVED = frozenset(
    {
        "about",
        "account",
        "advisories",
        "apps",
        "blog",
        "codespaces",
        "collections",
        "contact",
        "copilot",
        "customer-stories",
        "dashboard",
        "discussions",
        "education",
        "enterprise",
        "enterprises",
        "events",
        "explore",
        "features",
        "git-guides",
        "github-copilot",
        "home",
        "issues",
        "join",
        "login",
        "logout",
        "marketplace",
        "mobile",
        "new",
        "nonprofit",
        "notifications",
        "organizations",
        "orgs",
        "password_reset",
        "premium-support",
        "pricing",
        "pulls",
        "readme",
        "resources",
        "search",
        "security",
        "sessions",
        "settings",
        "signup",
        "site",
        "solutions",
        "sponsors",
        "stars",
        "team",
        "topics",
        "trending",
        "users",
        "watching",
    }
)
_GIST_RESERVED = frozenset({"auth", "discover", "forked", "join", "login", "mine", "search", "starred"})
_GIST_ID_RE = re.compile(r"^[0-9a-f]{20,}$")


def _doi_key(candidate: str) -> str | None:
    doi = candidate.strip()
    trimmed = _DOI_TRAILING_SEGMENT_RE.sub("", doi)  # old Wiley-style /abstract, /full, /pdf
    if _DOI_SHAPE_RE.match(trimmed):
        doi = trimmed
    while True:
        stripped = doi.rstrip(".,;:'\"/")
        for opener, closer in (("(", ")"), ("[", "]")):
            if stripped.endswith(closer) and stripped.count(opener) < stripped.count(closer):
                stripped = stripped[:-1]
        if stripped == doi:
            break
        doi = stripped
    return "doi:" + doi.lower() if _DOI_SHAPE_RE.match(doi) else None


def _is_se_host(host: str) -> bool:
    if host.endswith(".stackexchange.com"):
        return True
    if host == "stackexchange.com":
        return False  # the network hub has no questions of its own
    labels = host.split(".")
    while len(labels) > 2 and labels[0] in _SE_SUBSITE_LABELS:
        labels.pop(0)
    return ".".join(labels) in _SE_ROOTS


def _youtube_key(u: _URL, path: str) -> str | None:
    if u.host == "youtu.be":
        match = _YOUTU_BE_PATH_RE.match(path)
        return f"yt:{match.group(1)}" if match else None
    if path.rstrip("/") == "/watch":
        for name, value in parse_qsl(u.query):
            if name == "v" and _YOUTUBE_ID_RE.match(value):
                return f"yt:{value}"
        return None
    match = _YOUTUBE_PATH_RE.match(path)
    if match and match.group(1) != "videoseries":
        return f"yt:{match.group(1)}"
    return None


def _github_repo_key(path: str) -> str | None:
    segments = path.strip("/").split("/")
    if len(segments) != 2:
        return None
    owner, repo = segments
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    if (
        not _GITHUB_OWNER_RE.match(owner)
        or owner.lower() in _GITHUB_RESERVED
        or not _GITHUB_REPO_RE.match(repo)
        or repo in (".", "..")
    ):
        return None
    return f"gh:{owner.lower()}/{repo.lower()}"


def _pubmed_key(host: str, path: str) -> str | None:
    if host == "pubmed.ncbi.nlm.nih.gov":
        match = _PUBMED_PATH_RE.match(path)
        return f"pmid:{match.group(1)}" if match else None
    if host == "ncbi.nlm.nih.gov":
        match = _NCBI_PUBMED_PATH_RE.match(path)
        if match:
            return f"pmid:{match.group(1)}"
    if host == "europepmc.org":
        match = _EUROPEPMC_PATH_RE.match(path)
        if match:
            kind = "pmid" if match.group(1).lower() == "med" else "pmc"
            return f"{kind}:{match.group(2)}"
    # PMC articles: ncbi.nlm.nih.gov/pmc/articles/..., pmc.ncbi.nlm.nih.gov/articles/...,
    # europepmc.org/articles/...
    match = _PMC_ARTICLE_PATH_RE.match(path)
    if match and bool(match.group(1)) == (host == "ncbi.nlm.nih.gov"):
        return f"pmc:{match.group(2)}"
    return None


def identifier_key(url: str) -> str | None:
    """Identifier that merges one document across URL forms, e.g. ``arxiv:2401.01234``.

    Families: arXiv, DOI, PMID/PMC, Stack Exchange question, YouTube video, GitHub repo root.
    Returns None when the URL is not one of those (it then dedupes by canonical URL).
    """
    u = _parse(url)
    if u is None:
        return None
    host = u.host
    path = unquote(u.path)
    if host in _ARXIV_HOSTS or host.endswith(".arxiv.org"):
        match = _ARXIV_PATH_RE.match(path)
        return f"arxiv:{match.group('id').lower()}" if match else None
    if host in _DOI_HOSTS:
        match = _DOI_ORG_PATH_RE.match(path)
        return _doi_key(match.group(1)) if match else None
    if host in ("pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "europepmc.org"):
        return _pubmed_key(host, path)
    if _is_se_host(host):
        match = _SE_QUESTION_PATH_RE.match(path)
        return f"se:{host}:{match.group(1)}" if match else None
    if host in _YOUTUBE_HOSTS or host == "youtu.be":
        return _youtube_key(u, path)
    if host == "github.com":
        return _github_repo_key(path)
    if host == "link.springer.com":
        match = _SPRINGER_DOI_RE.match(path)
        return _doi_key(match.group(1)) if match else None
    match = _PUBLISHER_DOI_RE.search(path)
    return _doi_key(match.group(1)) if match else None


# --- hosts, registrable domains, site keys ---------------------------------------------------

# Second-level labels under which every name is a separate registrant (heuristic subset of the PSL).
_MULTIPART_SUFFIXES = frozenset(
    {
        # UK, Australia, New Zealand
        *("co.uk", "ac.uk", "gov.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk", "sch.uk", "nhs.uk"),
        *("com.au", "net.au", "org.au", "edu.au", "gov.au", "asn.au", "id.au"),
        *("co.nz", "ac.nz", "org.nz", "net.nz", "govt.nz", "school.nz"),
        # Asia
        *("co.jp", "ac.jp", "ne.jp", "or.jp", "go.jp", "ad.jp", "ed.jp", "gr.jp", "lg.jp"),
        *("co.kr", "ac.kr", "go.kr", "or.kr", "ne.kr", "re.kr"),
        *("com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn", "ac.cn"),
        *("com.hk", "edu.hk", "gov.hk", "org.hk", "com.tw", "edu.tw", "gov.tw", "org.tw"),
        *("com.sg", "edu.sg", "gov.sg", "com.my", "edu.my", "com.ph", "edu.ph"),
        *("co.id", "ac.id", "go.id", "or.id", "co.th", "ac.th", "go.th", "or.th", "com.vn", "edu.vn"),
        *("co.in", "net.in", "org.in", "ac.in", "gov.in", "edu.in", "res.in", "nic.in"),
        *("com.pk", "edu.pk", "gov.pk"),
        # Middle East, Africa
        *("ac.ir", "co.ir", "org.ir", "gov.ir", "net.ir", "sch.ir", "id.ir"),
        *("com.tr", "net.tr", "org.tr", "gov.tr", "edu.tr"),
        *("co.il", "ac.il", "org.il", "gov.il", "com.sa", "edu.sa", "com.eg", "edu.eg"),
        *("co.za", "ac.za", "gov.za", "org.za", "com.ng"),
        # Americas, Europe
        *("com.br", "net.br", "org.br", "gov.br", "edu.br", "com.mx", "gob.mx", "edu.mx", "org.mx"),
        *("com.ar", "gob.ar", "com.co", "com.pe"),
        *("co.at", "ac.at", "or.at", "gv.at", "com.ua", "org.ua", "gov.ua", "com.pl", "com.es", "com.gr"),
    }
)
# Shared-hosting suffixes where every subdomain is a different site.
_SHARED_HOSTING_SUFFIXES = frozenset(
    {
        "github.io",
        "gitlab.io",
        "codeberg.page",
        "readthedocs.io",
        "readthedocs.org",
        "netlify.app",
        "vercel.app",
        "pages.dev",
        "workers.dev",
        "blogspot.com",
        "wordpress.com",
        "substack.com",
        "herokuapp.com",
        "web.app",
        "firebaseapp.com",
        "fly.dev",
        "deno.dev",
        "onrender.com",
        "glitch.me",
        "surge.sh",
        "neocities.org",
        "tumblr.com",
        "appspot.com",
        "azurewebsites.net",
        "hf.space",
        "streamlit.app",
        "gitbook.io",
        "notion.site",
        "wixsite.com",
    }
)
_PUBLIC_SUFFIXES = _MULTIPART_SUFFIXES | _SHARED_HOSTING_SUFFIXES


def display_host(url: str) -> str:
    """Host for display: lowercased, without ``www.``/``m.``/``mobile.``/``amp.``. Empty if unparseable."""
    u = _parse(url)
    return u.host if u else ""


def registrable_domain(host: str) -> str:
    """eTLD+1 by heuristic (no PSL download). IPs and single-label hosts come back unchanged."""
    host = host.strip().lower().rstrip(".")
    if not host or host.startswith("[") or ":" in host or _IPV4_RE.match(host):
        return host
    labels = host.split(".")
    for i in range(1, len(labels)):  # longest matching suffix first
        if ".".join(labels[i:]) in _PUBLIC_SUFFIXES:
            return ".".join(labels[i - 1 :])
    return ".".join(labels[-2:])


def _github_owner(u: _URL) -> str | None:
    segments = [s for s in u.path.split("/") if s]
    if not segments or not _GITHUB_OWNER_RE.match(segments[0]):
        return None
    owner = segments[0].lower()
    if u.host == "github.com":
        return None if owner in _GITHUB_RESERVED else owner
    if u.host == "gist.github.com":
        if owner in _GIST_RESERVED or (len(segments) == 1 and _GIST_ID_RE.match(owner)):
            return None
        return owner
    return None


def site_key(url: str) -> str:
    """Diversity key: ``github.com/<owner>`` for GitHub and gist URLs, else the registrable domain."""
    u = _parse(url)
    if u is None:
        return ""
    owner = _github_owner(u)
    if owner:
        return f"github.com/{owner}"
    return registrable_domain(u.host)


def is_stackexchange(site_key: str) -> bool:
    """True for site keys on the Stack Exchange network (they get a larger diversity cap)."""
    key = site_key.strip().lower()
    return key in _SE_ROOTS or any(key.endswith("." + root) for root in _SE_ROOTS)


# --- merge -----------------------------------------------------------------------------------

_ID_FIRST = "abcdefghijklmnopqrstuvwxyz"
_ID_REST = "abcdefghijklmnopqrstuvwxyz0123456789"


def make_doc_id(key: str, taken: set[str]) -> str:
    """Opaque 4-char id (``[a-z][a-z0-9]{3}``) hashed from ``key``; rehashes with a salt on collision.

    The id is added to ``taken``. It grows past 4 chars only if ~1,000 salts all collide.
    """
    attempt = 0
    while True:
        material = key if attempt == 0 else f"{attempt}\x00{key}"
        digest = hashlib.blake2b(material.encode("utf-8", "surrogatepass"), digest_size=16).digest()
        number = int.from_bytes(digest, "big")
        chars = [_ID_FIRST[number % 26]]
        number //= 26
        for _ in range(3 + attempt // 1000):
            chars.append(_ID_REST[number % 36])
            number //= 36
        doc_id = "".join(chars)
        if doc_id not in taken:
            taken.add(doc_id)
            return doc_id
        attempt += 1


def _is_https(url: str) -> bool:
    return url.strip().lower().startswith("https://")


def _pick_url(raws: list[RawResult], canonicals: list[str]) -> str:
    """First-seen URL, upgraded to https when any raw is https (same URL form preferred)."""
    first = raws[0]
    if _is_https(first.url):
        return first.url
    https = [(r, c) for r, c in zip(raws, canonicals, strict=True) if _is_https(r.url)]
    if not https:
        return first.url
    for raw, canonical in https:
        if canonical == canonicals[0]:
            return raw.url
    return https[0][0].url


def _longest(texts: list[str]) -> str:
    return max(texts, key=lambda t: len(t.strip()))  # max() keeps the first on ties


def merge_results(raws: list[RawResult]) -> list[Doc]:
    """Merge raw rows from all SearXNG requests into Docs, in order of first appearance.

    Groups by identifier key when present, else by canonical URL. No near-duplicate title
    collapse. Ranks below 1 are treated as unknown and left out of ``request_ranks``.
    """
    groups: dict[str, list[tuple[RawResult, str, str | None]]] = {}
    for raw in raws:
        canonical = canonical_url(raw.url)
        ident = identifier_key(raw.url)
        key = f"id:{ident}" if ident else f"url:{canonical}"
        groups.setdefault(key, []).append((raw, canonical, ident))

    taken: set[str] = set()
    docs: list[Doc] = []
    for key, members in groups.items():
        rows = [raw for raw, _, _ in members]
        url = _pick_url(rows, [canonical for _, canonical, _ in members])
        ranks: dict[str, int] = {}
        scores: dict[str, float] = {}
        engines: set[str] = set()
        positions: list[int] = []
        for raw in rows:
            engines.update(raw.engines)
            positions.extend(raw.positions)
            if raw.rank >= 1:
                ranks[raw.request] = min(ranks.get(raw.request, raw.rank), raw.rank)
            scores[raw.request] = max(scores.get(raw.request, raw.score), raw.score)
        docs.append(
            Doc(
                id=make_doc_id(key, taken),
                url=url,
                canonical=canonical_url(url),
                ident=members[0][2],
                title=_longest([r.title for r in rows]),
                snippet=_longest([r.content for r in rows]),
                domain=display_host(url),
                site_key=site_key(url),
                engines=engines,
                category=next((r.category for r in rows if r.category is not None), None),
                published=next((r.published for r in rows if r.published is not None), None),
                request_ranks=ranks,
                request_scores=scores,
                positions=positions,
                from_vertical=any(r.request == "vertical" for r in rows),
            )
        )
    return docs


# --- post filters and diversity --------------------------------------------------------------

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://")


def _normalize_site(site: str) -> str:
    site = _SCHEME_RE.sub("", site.strip().lower())
    site = site.split("/", 1)[0].split(":", 1)[0].strip(".")
    site = site.removeprefix("*.")
    return site.removeprefix("www.") if site.count(".") >= 2 else site


def _doc_hosts(doc: Doc) -> set[str]:
    hosts = {doc.domain} if doc.domain else set()
    u = _parse(doc.url)
    if u is not None:
        hosts.add(u.host)
        hosts.add(u.raw_host.removeprefix("www.") if u.raw_host.count(".") >= 2 else u.raw_host)
    return hosts


def _on_any_site(hosts: set[str], sites: list[str]) -> bool:
    return any(host == site or host.endswith("." + site) for host in hosts for site in sites)


def apply_post_filters(docs: list[Doc], pq: ParsedQuery) -> list[Doc]:
    """Enforce ``site:``, ``-site:`` and ``-term`` for engines that ignore them. Order preserved.

    Sites match the host or any subdomain, ignoring ``www.``; a path in ``site:`` is ignored.
    Excluded terms match whole words, case-insensitively, in title, snippet and URL. Terms shorter
    than 3 characters are only forwarded to SearXNG, not post-filtered (`ls -la` must not drop "la").
    Quoted phrases are not filters.
    """
    sites = [s for s in map(_normalize_site, pq.sites) if s]
    excluded = [s for s in map(_normalize_site, pq.exclude_sites) if s]
    terms = [t.strip() for t in pq.exclude_terms if len(t.strip().strip('"')) >= 3]
    term_res = [re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.IGNORECASE) for t in terms]
    kept: list[Doc] = []
    for doc in docs:
        if sites or excluded:
            hosts = _doc_hosts(doc)
            if sites and not _on_any_site(hosts, sites):
                continue
            if excluded and _on_any_site(hosts, excluded):
                continue
        if term_res:
            haystack = f"{doc.title}\n{doc.snippet}\n{unquote(doc.url)}"
            if any(rx.search(haystack) for rx in term_res):
                continue
        kept.append(doc)
    return kept


def diversify(
    docs: list[Doc], k: int, *, waive: bool = False, per_site: int = 2, per_site_se: int = 3
) -> list[Doc]:
    """Cap docs per ``site_key`` inside the first ``k`` slots; return a full permutation.

    Over-cap docs move behind the capped head in their original order, so ``result[:k]`` is
    backfilled from them when the head is short. ``waive`` keeps the input order.
    """
    if waive:
        return list(docs)
    head: list[Doc] = []
    tail: list[Doc] = []
    counts: dict[str, int] = {}
    for doc in docs:
        cap = per_site_se if is_stackexchange(doc.site_key) else per_site
        used = counts.get(doc.site_key, 0)
        if len(head) < k and used < cap:
            head.append(doc)
            counts[doc.site_key] = used + 1
        else:
            tail.append(doc)
    return head + tail


# --------------------------------------------------------------------------------------------- reposts
# Syndicated copies of one article on different sites (a blog post reposted on dev.to or Medium) have
# different URLs and no shared identifier, so URL/identifier merging misses them. Two results are one
# article when (1) their sites differ, (2) their normalised titles are equal and specific (at least
# three content words that are not generic page words), and (3) their snippets are near-duplicates:
# word 3-shingle containment |A ∩ B| / min(|A|, |B|) ≥ REPOST_CONTAINMENT, with at least
# MIN_SHINGLES shingles in the shorter snippet. Reposts copy text verbatim, so containment stays high
# even when engines cut the snippet at different points; two different articles that share a title
# share few 3-word sequences. Version pages ("What's new in Python 3.11" vs "3.12") differ in the title
# and never merge; short or generic titles ("Getting Started", "Introduction") never merge.
REPOST_CONTAINMENT = 0.5
MIN_SHINGLES = 5
_TITLE_SEPARATORS = re.compile(r"\s+(?:-|\||–|—|·|•)\s+")
_NON_ALNUM = re.compile(r"[^\w]+", re.UNICODE)
_GENERIC_TITLE_WORDS = frozenset(
    [
        "getting",
        "started",
        "start",
        "quick",
        "quickstart",
        "guide",
        "guides",
        "introduction",
        "intro",
        "overview",
        "documentation",
        "docs",
        "doc",
        "home",
        "homepage",
        "page",
        "welcome",
        "tutorial",
        "tutorials",
        "reference",
        "api",
        "index",
        "about",
        "contact",
        "blog",
        "news",
        "faq",
        "help",
        "support",
        "download",
        "downloads",
        "installation",
        "install",
        "setup",
        "release",
        "releases",
        "notes",
        "changelog",
        "features",
        "pricing",
        "login",
        "sign",
        "signup",
        "search",
        "results",
        "chapter",
        "part",
        "section",
        "examples",
        "example",
    ]
)
_TITLE_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "by",
        "from",
        "at",
        "is",
        "are",
        "how",
        "what",
        "why",
        "your",
        "you",
    ]
)


def _norm_words(text: str) -> list[str]:
    return [w for w in _NON_ALNUM.sub(" ", text.casefold()).split() if w]


def repost_title_key(title: str) -> str | None:
    """Normalised article title (site-name segments removed), or None if too generic to compare."""
    segments = [s for s in _TITLE_SEPARATORS.split(title or "") if s.strip()]
    if not segments:
        return None
    main = max(segments, key=len)
    words = _norm_words(main)
    content = [w for w in words if len(w) >= 3 and w not in _TITLE_STOP]
    if len(content) < 3 or all(w in _GENERIC_TITLE_WORDS for w in content):
        return None
    return " ".join(words)


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = _norm_words(text)
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def snippet_containment(a: str, b: str) -> float | None:
    sa, sb = _shingles(a), _shingles(b)
    smaller = min(len(sa), len(sb))
    if smaller < MIN_SHINGLES:
        return None
    return len(sa & sb) / smaller


def is_repost(a: Doc, b: Doc) -> bool:
    if registrable_domain(a.domain) == registrable_domain(b.domain):
        return False
    ka, kb = repost_title_key(a.title), repost_title_key(b.title)
    if ka is None or ka != kb:
        return False
    c = snippet_containment(a.snippet, b.snippet)
    return c is not None and c >= REPOST_CONTAINMENT


def dedupe_reposts(ranked: list[Doc]) -> list[Doc]:
    """Fold cross-site reposts into the better-ranked copy (input order = rank). The kept doc gains the
    other's engines and request ranks, and lists the dropped URL in `duplicates`."""
    kept: list[Doc] = []
    by_title: dict[str, list[Doc]] = {}
    for d in ranked:
        key = repost_title_key(d.title)
        twin = next((k for k in by_title.get(key, []) if is_repost(k, d)), None) if key else None
        if twin is None:
            kept.append(d)
            if key:
                by_title.setdefault(key, []).append(d)
            continue
        twin.engines |= d.engines
        for label, rank in d.request_ranks.items():
            twin.request_ranks[label] = min(rank, twin.request_ranks.get(label, rank))
        twin.duplicates.append(d.url)
        if "repost" not in twin.flags:
            twin.flags.append("repost")
    return kept
