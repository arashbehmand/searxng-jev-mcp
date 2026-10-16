"""Stage A: query handling.

:func:`parse_query` turns a raw query into a :class:`ParsedQuery`: search operators are parsed
out, private data is scrubbed (paths, usernames, emails, ids, timestamps, secrets, IPs), the text
is capped, and script, language and recency cues are detected. :func:`to_searxng_q` reassembles
the query for SearXNG and neutralizes SearXNG's own in-query syntax (``!bang``, ``:lang``,
``<timeout``). :func:`keyword_form` builds the short keyword query for vertical engines.

Everything is pure and deterministic, except that the OS username is read once and cached.
"""

from __future__ import annotations

import datetime as dt
import functools
import getpass
import itertools
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from wordfreq import zipf_frequency

from searxng_jev_mcp.models import ParsedQuery, Script

__all__ = [
    "CLI_COMMANDS",
    "FLAG_WORDS",
    "STOPWORDS",
    "TIME_RANGES",
    "detect_script",
    "escape_searxng_syntax",
    "infer_language",
    "keyword_form",
    "parse_query",
    "scrub",
    "time_range_cue",
    "to_searxng_q",
]

TIME_RANGES: frozenset[str] = frozenset({"day", "week", "month", "year"})
_RANGE_WIDTH = {"day": 0, "week": 1, "month": 2, "year": 3}

# --------------------------------------------------------------------------- normalization

# Zero-width and bidi controls (ZWSP, ZWJ, LRM/RLM, embeddings/overrides/isolates, word joiner,
# BOM, soft hyphen, Arabic letter mark). ZWNJ (U+200C) is kept between letters: it is part of
# Persian spelling (e.g. "می‌خواهم") and documents contain it too.
_INVISIBLE_RE = re.compile(
    r"[\u00ad\u061c\u180e\u200b\u200d-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]"
)
_STRAY_ZWNJ_RE = re.compile(r"(?<!\w)\u200c+|\u200c+(?!\w)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\uff02": '"'})


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _STRAY_ZWNJ_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text).translate(_QUOTES)
    return " ".join(text.split())


# --------------------------------------------------------------------------- operators

_PHRASE_RE = re.compile(r'(?<!\S)(-?)"([^"]*)"')
_OPERATOR_RE = re.compile(r"^(-?)(site|filetype|ext):(.+)$", re.IGNORECASE)
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_PORT_RE = re.compile(r":\d*$")
_HOST_RE = re.compile(r"[\w-]+(?:\.[\w-]+)*")
_IPV4_HOST_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
_FILETYPE_RE = re.compile(r"[a-z0-9]{1,10}")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

_LEAD_PUNCT = "\"'“”‘’([{¿¡«"
_TRAIL_PUNCT = "\"'“”‘’?!,;:.»؟،"
_CLOSERS = {")": "(", "]": "[", "}": "{"}


@dataclass(slots=True)
class _Parts:
    words: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=list)
    exclude_sites: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    filetypes: list[str] = field(default_factory=list)
    kinds: set[str] = field(default_factory=set)


def _clean_token(token: str) -> str:
    """Strip surrounding punctuation, keeping balanced brackets (``foo.bar()``) and ``C++``/``.NET``."""
    tok = token.lstrip(_LEAD_PUNCT).rstrip(_TRAIL_PUNCT)
    while tok and tok[-1] in _CLOSERS and tok.count(_CLOSERS[tok[-1]]) < tok.count(tok[-1]):
        tok = tok[:-1].rstrip(_TRAIL_PUNCT)
    return tok


def _normalize_site(value: str) -> str | None:
    value = _SCHEME_RE.sub("", value.strip().rstrip(",;"))
    host, _, path = value.partition("/")
    host = _PORT_RE.sub("", host.lower()).rstrip(".")
    host = host.removeprefix("*.").removeprefix("www.")
    if not _HOST_RE.fullmatch(host):
        return None
    path = path.rstrip("/")
    return f"{host}/{path}" if path else host


def _normalize_filetype(value: str) -> str | None:
    ext = value.strip().lower().lstrip(".").rstrip(",;")
    return ext if _FILETYPE_RE.fullmatch(ext) else None


# Command words: a query containing one of these as a standalone token is about a command line, so
# its "-x" tokens are flags, never exclusions. Excluding "name" from `find . -name "*.py"` would make
# the engines (and the local post-filter) drop exactly the pages that answer the query.
_CLI_COMMANDS_TEXT = """
find tar grep egrep fgrep rg ls cp mv rm chmod chown chgrp ssh scp sftp rsync curl wget docker
docker-compose podman kubectl helm minikube git gh glab svn npm npx yarn pnpm bunx pip pip3 pipx python
python2 python3 ipython jupyter node deno cargo rustc rustup go java javac mvn gradle dotnet gcc g++
clang clang++ make cmake ps kill du df sed awk xargs ssh-keygen ssh-add ssh-copy-id openssl systemctl
journalctl launchctl brew apt apt-get dpkg rpm yum dnf pacman zypper flatpak uv uvx poetry pdm conda
pyenv nvm pytest mypy ruff tsc ffmpeg zip unzip gzip gunzip bzip2 xz 7z unrar sudo sh bash zsh fish
nohup nc nmap dig ping lsof netstat ifconfig iptables tail head cat less sort uniq wc tee touch mkdir ln
mount umount crontab gpg jq terraform ansible ansible-playbook vagrant aws gcloud az psql mysql sqlite3
tmux vim nvim strace dmesg sysctl uname useradd usermod lsblk fdisk mkfs sha256sum md5sum xcode-select
xcrun codesign diskutil yt-dlp pandoc powershell pwsh winget choco robocopy xcopy
"""
CLI_COMMANDS: frozenset[str] = frozenset(_CLI_COMMANDS_TEXT.split())

# "-word" tokens that are flags even without a command word in the query: ls/tar/rm/grep/ps/rsync
# clusters, `set -euo`, find/test predicates, common long options typed with one dash, and PowerShell.
_FLAG_WORDS_TEXT = """
la al lh lah lha alh lt ltr lrt lart latr rf fr rfv rp rn ri rni rin irn inr rnw xvf xzvf xvzf zxvf
czvf cvzf cvf czf xzf zxf xjf xjvf cjf cjvf tvf tzvf xf cf tf it ti euo eux aux ef avz avzp azv avh
ivh uvh exec execdir name iname ipath type print fprint delete mtime atime ctime mmin amin cmin size
user group maxdepth mindepth path newer perm depth empty prune xdev help version verbose force
recursive recurse quiet all noprofile nologo noninteractive executionpolicy erroraction
interactive invert vcodec acodec preset
"""
FLAG_WORDS: frozenset[str] = frozenset(_FLAG_WORDS_TEXT.split())

_DASH_WORD_RE = re.compile(r"--?[^\W_]")  # "-x..." or "--x...": a flag or an exclusion
_LONG_FLAG_RE = re.compile(r"--[^\W_]")
_SHORT_FLAG_RE = re.compile(r"-[A-Za-z]")
_EXCLUSION_TERM_RE = re.compile(r"[^\W\d_]+(?:[-\u200c][^\W\d_]+)*")  # letters; inner "-" or ZWNJ
_OPERATOR_LIKE_RE = re.compile(r"-[^\W\d_]+:\S")  # -intitle:foo, -inurl:bar: passed through
# PowerShell cmdlets (Get-ChildItem, Test-Path): their "-Filter"/"-Path" are parameters.
_CMDLET_RE = re.compile(
    r"(?:Get|Set|New|Remove|Add|Invoke|Start|Stop|Restart|Test|Write|Import|Export|Select|Where|ForEach"
    r"|Out|Format|ConvertTo|ConvertFrom|Copy|Move|Rename|Clear|Install|Uninstall|Update|Enable|Disable"
    r"|Register|Unregister|Measure|Resolve|Expand|Compress|Mount|Dismount)-[A-Z][A-Za-z0-9]{3,}"
)


def _cli_context(tokens: Iterable[str]) -> str | None:
    """How sure we are that the query is a command line.

    "strong": an unmistakable flag (``--long`` or a single letter ``-x``): every dash token is a flag.
    "weak": a command word starts the query or directly precedes a dash token (``tar -xvf``): only
    flag-looking dash tokens (short clusters, digits, known flag words) are flags; ``python -numpy``
    stays an exclusion. None: plain query ("how to make bread -yeast" excludes yeast).
    """
    toks = list(tokens)
    words = [_clean_token(t) for t in toks]
    weak = False
    for i, token in enumerate(toks):
        if token.startswith("-"):
            if _LONG_FLAG_RE.match(token) or _SHORT_FLAG_RE.fullmatch(words[i]):
                return "strong"
            continue
        if _CMDLET_RE.fullmatch(words[i]):
            return "strong"
        if words[i].lower() in CLI_COMMANDS and (
            i == 0 or (i + 1 < len(toks) and toks[i + 1].startswith("-"))
        ):
            weak = True
    return "weak" if weak else None


def _flag_like(term: str) -> bool:
    return len(term) <= 3 or term.lower() in FLAG_WORDS or not term.isalpha()


def _is_exclusion_term(term: str) -> bool:
    """A real "-term" exclusion: letters only (inner hyphens allowed), >= 3 chars, not a flag word."""
    return len(term) >= 3 and bool(_EXCLUSION_TERM_RE.fullmatch(term)) and term.lower() not in FLAG_WORDS


def _take_token(token: str, parts: _Parts, cli: str | None) -> str | None:
    """Record ``token`` if it is an operator (return None), else return the text to keep.

    A dash token that is not an exclusion is a flag and is kept without its dashes (``-xvf`` →
    ``xvf``, ``--force`` → ``force``): engines like Bing read ``-xvf`` as "NOT xvf".
    """
    m = _OPERATOR_RE.match(token)
    if m:
        negated, name, value = m.group(1) == "-", m.group(2).lower(), m.group(3)
        if name == "site":
            site = _normalize_site(value)
            if site is None:
                return token
            if _IPV4_HOST_RE.fullmatch(site.split("/", 1)[0]):
                parts.kinds.add("ip")  # never send a (likely private) IP to the engines
                return None
            (parts.exclude_sites if negated else parts.sites).append(site)
            return None
        if negated:  # -filetype:pdf has no portable meaning: pass it through as an exclusion
            parts.exclude_terms.append(_clean_token(token[1:]))
            return None
        ext = _normalize_filetype(value)
        if ext is None:
            return token
        parts.filetypes.append(ext)
        return None
    if not _DASH_WORD_RE.match(token):
        return token
    term = _clean_token(token[1:])
    if _NUMBER_RE.fullmatch(term):
        return token  # "x -1", "temperature -3.5": a number, keep it as typed
    as_flag = cli == "strong" or (cli == "weak" and _flag_like(term))
    if not as_flag and (_is_exclusion_term(term) or _OPERATOR_LIKE_RE.match(token)):
        parts.exclude_terms.append(term)
        return None
    return token[2:] if token.startswith("--") else token[1:]


def _split_operators(text: str) -> _Parts:
    parts = _Parts()

    def take_phrase(m: re.Match[str]) -> str:
        body = " ".join(m.group(2).split())
        if body:
            (parts.exclude_terms if m.group(1) else parts.phrases).append(body)
        return " "

    tokens = _PHRASE_RE.sub(take_phrase, text).split()
    cli = _cli_context(itertools.chain(tokens, *(p.split() for p in parts.phrases)))
    for token in tokens:
        kept = _take_token(token, parts, cli)
        if kept is not None:
            parts.words.append(kept)
    return parts


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _negate(term: str) -> str:
    return f'-"{term}"' if any(ch.isspace() for ch in term) else f"-{term}"


def _operator_parts(pq: ParsedQuery | _Parts) -> list[str]:
    return [
        *(f"site:{s}" for s in pq.sites),
        *(f"-site:{s}" for s in pq.exclude_sites),
        *(_negate(t) for t in pq.exclude_terms),
        *(f"filetype:{f}" for f in pq.filetypes),
    ]


# --------------------------------------------------------------------------- SearXNG syntax


def _escape_token(token: str) -> str:
    head, rest = token[0], token[1:]
    if head == "<":
        # SearXNG: "<N" with N.isdigit() sets a timeout and is dropped from the query.
        return f"< {rest}" if rest.isdigit() else token
    if head == ":":
        # SearXNG: ":xx" sets the language when xx is a code/name/country ("-"/"_" are separators).
        return f": {rest}" if rest.replace("-", "").replace("_", "").isalpha() else token
    if head == "!":
        # SearXNG: "!x" / "!!x" pick engines or redirect; a token of only "!" is never consumed.
        remainder = token.lstrip("!")
        return f"! {_escape_token(remainder)}" if remainder else "!"
    return token


def escape_searxng_syntax(q: str) -> str:
    """Neutralize SearXNG's per-token syntax by inserting a space after the sigil.

    ``pandas <2`` → ``pandas < 2``, ``:fr`` → ``: fr``, ``!!w`` → ``! w``. Idempotent. Applies
    inside quoted phrases too, because SearXNG splits the whole ``q`` on whitespace.
    """
    return " ".join(_escape_token(tok) for tok in q.split())


def to_searxng_q(pq: ParsedQuery) -> str:
    """Assemble ``q`` for SearXNG: text, phrases, then operators; escaped unless ``pq.raw``."""
    parts = [pq.text, *(f'"{p}"' for p in pq.phrases), *_operator_parts(pq)]
    q = " ".join(p for p in parts if p)
    return q if pq.raw else escape_searxng_syntax(q)


# --------------------------------------------------------------------------- scrub

_Repl = Callable[[re.Match[str]], str]

_PEM_RE = re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END[A-Z ]*PRIVATE KEY-----|\Z)")
_URL_USERINFO_RE = re.compile(r"(?<=://)[^\s/@:]+:[^\s/@]+@")
_KV_SECRET_RE = re.compile(
    r"(?P<key>(?<![\w-])[\w-]*?(?:api[_-]?key|apikey|token|secret|passw(?:or)?d|pwd)[\"']?)"
    r"\s*(?P<sep>[=:])\s*[\"']?(?P<val>[^\s\"'&,;]+)[\"']?",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\b(bearer)\s+(?=[\w.~+/-]*\d)[\w.~+/-]{12,}=*", re.IGNORECASE)
_KNOWN_TOKEN_RE = re.compile(
    r"(?<![\w-])(?:"
    r"sk-(?=[\w-]*\d)[\w-]{20,}"  # OpenAI / OpenRouter / Anthropic style
    r"|gh[pousr]_[A-Za-z0-9]{20,}"  # GitHub tokens
    r"|github_pat_\w{20,}"
    r"|xox[abposr]-[A-Za-z0-9-]{10,}"  # Slack
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Za-z])"  # AWS access key id
    r"|AIza[\w-]{35}"  # Google API key
    r"|[rs]k_(?:live|test)_[0-9A-Za-z]{10,}"  # Stripe
    r"|hf_[A-Za-z0-9]{30,}"  # Hugging Face
    r"|glpat-[\w-]{20,}"  # GitLab
    r"|eyJ[\w-]{5,}\.eyJ[\w-]{5,}(?:\.[\w-]*)?"  # JWT
    r")"
)
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])(?P<local>[\w.+-]+)@"
    r"(?:[\w-]+(?:\.[\w-]+)*\.[^\W\d_]{2,}|\d{1,3}(?:\.\d{1,3}){3})(?![\w-])"
)
# Only paths that can identify the user (home directories) or are random temp paths are scrubbed.
# System paths (/dev/null, /etc/fstab, /usr/local/bin, C:\Windows\System32) are search terms.
_PATH_SEGMENT = r"[^\s/\\\"'<>|:;]+"  # one segment: the user name, a temp dir
_PATH_REST = r"(?:(?!:/)[^\s\"'<>|;])*"  # the rest, up to a ":/" PATH-list separator
_UNIX_PRIVATE_DIRS = (
    rf"(?:Users|home|root|tmp|private/tmp|var/folders|private/var/folders|mnt/[a-z]/Users)/{_PATH_SEGMENT}"
)
_WINDOWS_HOME = rf"[A-Za-z]:[\\/]+Users[\\/]+{_PATH_SEGMENT}"
_PATH_RE = re.compile(
    rf"\bfile://(?:localhost)?/(?:{_UNIX_PRIVATE_DIRS}|{_WINDOWS_HOME}){_PATH_REST}"  # file:///Users/bob/x
    rf"|(?<![\w\\/]){_WINDOWS_HOME}{_PATH_REST}"  # C:\Users\bob\..., C:/Users/bob/...
    rf"|(?<![\w\\/])\\\\[\w.$-]+\\{_PATH_REST}"  # \\server\share\...: names an internal host
    rf"|(?<![\w.~/\\])~[\w.-]*/{_PATH_REST}"  # ~/..., ~bob/...
    rf"|(?<![\w.~/\\])/{_UNIX_PRIVATE_DIRS}{_PATH_REST}",  # /Users/bob/..., /tmp/x, /var/folders/...
    re.IGNORECASE,
)
_PATH_TRAIL = ".,;:!?)]}"
_LINE_SUFFIX_RE = re.compile(r"(?::\d+)+$")
_FILE_LIKE_RE = re.compile(r"(?:\w[\w+-]*)?(?:\.\w[\w+-]*)*\.[A-Za-z0-9]{1,10}")
_UUID_RE = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}(?![0-9A-Fa-f])")
_TIME = r"\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
_DATETIME = rf"\d{{4}}-\d{{2}}-\d{{2}}[T ]{_TIME}"
_TIMESTAMP_RE = re.compile(
    rf"\[\s*{_DATETIME}\s*\]"  # [2026-10-08 14:03:22]
    rf"|(?<!\d){_DATETIME}(?![\d:])"  # 2026-10-08T14:03:22Z
    r"|(?<![\d:.])\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?(?![\d:])"  # 14:03:22.123
)
_IPV4_RE = re.compile(
    r"(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]{1,5})?(?:/[0-9]{1,2})?(?!\w)(?!\.\d)"
)
_HEXGROUP = r"[0-9A-Fa-f]{1,4}"
_IPV6_RE = re.compile(
    rf"(?<![\w:])(?:(?:{_HEXGROUP}:){{7}}{_HEXGROUP}"
    rf"|(?:{_HEXGROUP}:){{1,6}}:(?:{_HEXGROUP}(?::{_HEXGROUP}){{0,5}})?)(?![\w:])"
)
_HEX_RE = re.compile(r"(?<!\w)(?:0[xX](?P<addr>[0-9A-Fa-f]{6,})|[0-9A-Fa-f]{32,})(?!\w)")
_B64_RE = re.compile(r"(?<![\w+/=-])[A-Za-z0-9+/_-]{40,}={0,2}(?![\w+/=-])")
_ORPHAN_RE = re.compile(r"(?<!\S)(?:\(\)|\[\]|\{\}|<>|\"\"|''|``)(?!\S)")

_GENERIC_MAIL_LOCALS = frozenset({"git", "noreply", "no-reply"})
# OS accounts that are also ordinary words; scrubbing them would damage queries for no privacy gain.
_GENERIC_USERNAMES_TEXT = """admin administrator app build builder ci codespace debian default dev developer
docker ec2-user guest home jenkins node nobody root runner test tester ubuntu user users vagrant"""
_GENERIC_USERNAMES = frozenset(_GENERIC_USERNAMES_TEXT.split())


def _blank(_m: re.Match[str]) -> str:
    return " "


def _remove(_m: re.Match[str]) -> str:
    return ""


def _kv_repl(m: re.Match[str]) -> str:
    val = m.group("val")
    # "password: how do I reset it" is prose, so the ":" form needs a long value with a digit.
    secret = len(val) >= 3 if m.group("sep") == "=" else len(val) >= 8 and any(c.isdigit() for c in val)
    return m.group("key") if secret else m.group(0)


def _bearer_repl(m: re.Match[str]) -> str:
    return m.group(1)


def _email_repl(m: re.Match[str]) -> str:
    return m.group(0) if m.group("local").lower() in _GENERIC_MAIL_LOCALS else " "


def _path_repl(m: re.Match[str]) -> str:
    """Replace a path by its basename when that looks like a file (``main.py``), else drop it."""
    match = m.group(0)
    core = match.rstrip(_PATH_TRAIL)
    tail = match[len(core) :]
    name = _LINE_SUFFIX_RE.sub("", re.split(r"[\\/]", core)[-1])
    if _FILE_LIKE_RE.fullmatch(name):
        return f"{name}{tail}"
    return tail or " "


def _ipv4_repl(m: re.Match[str]) -> str:
    address = m.group(0).split(":", 1)[0].split("/", 1)[0]
    return " " if all(int(octet) <= 255 for octet in address.split(".")) else m.group(0)


def _ipv6_repl(m: re.Match[str]) -> str:
    text = m.group(0)
    groups = [g for g in text.split(":") if g]
    # ">= 3 groups with a digit" keeps "::1", "fe80::1"-like stubs and C++ "a::b" out.
    return " " if len(groups) >= 3 and any(c.isdigit() for c in text) else text


def _hex_repl(m: re.Match[str]) -> str:
    addr = m.group("addr")
    if addr is not None and len(addr) == 8:  # 0x80070005: a Windows HRESULT, keep it
        return m.group(0)
    return " "


def _char_class(ch: str) -> int:
    if ch.isdigit():
        return 0
    if ch.isupper():
        return 1
    if ch.islower():
        return 2
    return 3


def _looks_random(run: str) -> bool:
    """Random token vs slug/identifier: mixed case, scattered digits, few separators, and the
    character class (digit/upper/lower) changes often. Passes ~97% of random base64 runs."""
    core = run.rstrip("=")
    alnum = [c for c in core if c.isalnum()]
    if len(alnum) < 2 or core.lower() == core or core.upper() == core:
        return False
    digit_runs = sum(1 for is_digit, _ in itertools.groupby(alnum, key=str.isdigit) if is_digit)
    separators = (len(core) - len(alnum)) / len(core)
    changes = sum(_char_class(a) != _char_class(b) for a, b in itertools.pairwise(alnum))
    return digit_runs >= 2 and separators <= 0.12 and changes / (len(alnum) - 1) >= 0.42


def _b64_repl(m: re.Match[str]) -> str:
    return " " if _looks_random(m.group(0)) else m.group(0)


_RULES: tuple[tuple[str, re.Pattern[str], _Repl], ...] = (
    ("secret", _PEM_RE, _blank),
    ("secret", _URL_USERINFO_RE, _remove),
    ("secret", _KV_SECRET_RE, _kv_repl),
    ("secret", _BEARER_RE, _bearer_repl),
    ("secret", _KNOWN_TOKEN_RE, _blank),
    ("email", _EMAIL_RE, _email_repl),
    ("path", _PATH_RE, _path_repl),
    ("uuid", _UUID_RE, _blank),
    ("timestamp", _TIMESTAMP_RE, _blank),
    ("ip", _IPV4_RE, _ipv4_repl),
    ("ip", _IPV6_RE, _ipv6_repl),
    ("hex", _HEX_RE, _hex_repl),
    ("secret", _B64_RE, _b64_repl),
)


@functools.cache
def _os_username() -> str | None:
    """The current OS username if it is worth scrubbing (>= 3 chars, not a generic account)."""
    try:
        name = getpass.getuser().strip()
    except Exception:
        return None
    if len(name) < 3 or name.lower() in _GENERIC_USERNAMES:
        return None
    return name


@functools.cache
def _user_at_host_re(name: str) -> re.Pattern[str]:
    """``alice@devbox`` / ``alice@devbox:/srv`` (ssh, scp) for the OS username. The bare name is
    not scrubbed: it is often an ordinary word, and paths that contain it are scrubbed anyway."""
    return re.compile(rf"(?<![\w.+-]){re.escape(name)}@[\w.-]*\w(?::\S*)?", re.IGNORECASE)


def _apply(text: str, kind: str, pattern: re.Pattern[str], repl: _Repl, kinds: set[str]) -> str:
    def sub(m: re.Match[str]) -> str:
        out = repl(m)
        if out != m.group(0):
            kinds.add(kind)
        return out

    return pattern.sub(sub, text)


def scrub(text: str) -> tuple[str, list[str]]:
    """Remove private data before the query leaves the machine.

    Returns the cleaned text (whitespace collapsed) and the sorted distinct kinds removed:
    ``path``, ``username``, ``email``, ``uuid``, ``hex``, ``timestamp``, ``secret``, ``ip``.
    Only home-directory and temp paths count as ``path`` (a file-like basename is kept); system
    paths, plain dates (``2026-10-08``), years, versions and error codes are kept. The OS username
    is removed only in ``user@host`` tokens, not as a standalone word.
    """
    kinds: set[str] = set()
    for kind, pattern, repl in _RULES:
        text = _apply(text, kind, pattern, repl, kinds)
    user = _os_username()
    if user:
        text = _apply(text, "username", _user_at_host_re(user), _blank, kinds)
    if kinds:
        text = _ORPHAN_RE.sub(" ", text)
    return " ".join(text.split()), sorted(kinds)


# --------------------------------------------------------------------------- length cap


def _assembled_len(text: str, phrases: list[str], ops_len: int) -> int:
    lengths = ([len(text)] if text else []) + [len(p) + 2 for p in phrases] + ([ops_len] if ops_len else [])
    return sum(lengths) + max(len(lengths) - 1, 0)


def _cut_words(text: str, room: int) -> str:
    if room <= 0:
        return ""
    if len(text) <= room:
        return text
    if text[room].isspace():
        return text[:room].rstrip()
    head = text[:room]
    space = head.rfind(" ")
    return head[:space].rstrip() if space > 0 else head  # no boundary (e.g. CJK): hard cut


def _cap(text: str, phrases: list[str], ops_len: int, max_chars: int) -> tuple[str, list[str], bool]:
    """Fit text + phrases + operators into ``max_chars``; operators are never cut."""
    if max_chars <= 0 or _assembled_len(text, phrases, ops_len) <= max_chars:
        return text, phrases, False
    kept = list(phrases)
    while kept and _assembled_len("", kept, ops_len) > max_chars:
        last = kept.pop()
        sep = 1 if kept or ops_len else 0
        short = _cut_words(last, max_chars - _assembled_len("", kept, ops_len) - sep - 2)
        if short:
            kept.append(short)
            break
    sep = 1 if kept or ops_len else 0
    return _cut_words(text, max_chars - _assembled_len("", kept, ops_len) - sep), kept, True


# --------------------------------------------------------------------------- script / language

_SCRIPT_RANGES: tuple[tuple[int, int, Script], ...] = (
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0x08A0, 0x08FF, "arabic"),
    (0xFB50, 0xFDFF, "arabic"),
    (0xFE70, 0xFEFF, "arabic"),
    (0x0400, 0x052F, "cyrillic"),
    (0x1100, 0x11FF, "cjk"),  # hangul jamo
    (0x3040, 0x30FF, "cjk"),  # hiragana, katakana
    (0x3130, 0x318F, "cjk"),  # hangul compatibility jamo
    (0x31F0, 0x31FF, "cjk"),  # katakana extensions
    (0x3400, 0x4DBF, "cjk"),
    (0x4E00, 0x9FFF, "cjk"),
    (0xAC00, 0xD7AF, "cjk"),  # hangul syllables
    (0xF900, 0xFAFF, "cjk"),
    (0xFF66, 0xFF9F, "cjk"),  # halfwidth katakana
    (0x20000, 0x323AF, "cjk"),
)
_LATIN_RANGES = ((0x0000, 0x024F), (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF), (0xFF21, 0xFF5A))
_KANA_RANGES = ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F))
_HANGUL_RANGES = ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))
_SCRIPT_ORDER: tuple[Script, ...] = ("arabic", "cjk", "cyrillic", "other", "latin")
_NON_LATIN_SHARE = 0.4

# پ چ ژ گ and the Persian forms of ک / ی
_PERSIAN_CHARS = frozenset("\u067e\u0686\u0698\u06af\u06a9\u06cc")
# ٹ ڈ ڑ ں ے: Urdu-only letters (Urdu also uses the Persian letters, so "fa" would be wrong)
_URDU_CHARS = frozenset("\u0679\u0688\u0691\u06ba\u06d2")
_UKRAINIAN_CHARS = frozenset("іїєґІЇЄҐ")


def _in_ranges(ch: str, ranges: Iterable[tuple[int, int]]) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def _script_of(ch: str) -> Script:
    cp = ord(ch)
    for lo, hi, script in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return script
    return "latin" if _in_ranges(ch, _LATIN_RANGES) else "other"


def detect_script(text: str) -> Script:
    """Dominant script by letter count; a non-Latin script with >= 40% of letters wins."""
    counts: dict[Script, int] = {}
    for ch in text:
        if ch.isalpha():
            script = _script_of(ch)
            counts[script] = counts.get(script, 0) + 1
    total = sum(counts.values())
    if not total:
        return "other"
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], _SCRIPT_ORDER.index(kv[0])))
    for script, n in ranked:
        if script != "latin" and n / total >= _NON_LATIN_SHARE:
            return script
    return ranked[0][0]


def infer_language(text: str, script: Script) -> str | None:
    """Language code when the script makes it unambiguous, else None."""
    chars = set(text)
    if script == "arabic":
        if chars & _URDU_CHARS:
            return None
        return "fa" if chars & _PERSIAN_CHARS else "ar"
    if script == "cjk":
        if any(_in_ranges(c, _KANA_RANGES) for c in text):
            return "ja"
        if any(_in_ranges(c, _HANGUL_RANGES) for c in text):
            return "ko"
        return "zh"
    if script == "cyrillic":
        return "uk" if chars & _UKRAINIAN_CHARS else "ru"
    return None


# --------------------------------------------------------------------------- time range


def _cue(*alternatives: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


_RANGE_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("day", _cue(r"(?<!usa )today", r"tonight", r"this\s+morning", r"(?:last|past)\s+24\s+hours", "امروز")),
    (
        "week",
        _cue(r"(?:this|past|last)\s+week", r"(?:last|past)\s+(?:7|seven)\s+days", r"این\s+هفته"),
    ),
    (
        "month",
        _cue(r"(?:this|past|last)\s+month", r"(?:last|past)\s+(?:30|thirty)\s+days", r"این\s+ماه"),
    ),
    ("year", _cue(r"(?:this|past)\s+year", r"(?:last|past)\s+12\s+months", "امسال")),
)
_RECENCY_RE = _cue(
    "latest", "newest", "recent", "recently", r"up[\s-]+to[\s-]+date", r"current\s+version", "جدید", "آخرین"
)
_YEAR_RE = re.compile(r"(?<![\w.:/-])\d{4}(?![\w:/-])(?!\.\d)")


def time_range_cue(text: str, today: dt.date | None = None) -> tuple[str | None, bool]:
    """(time_range a cue implies, recency word present). Diagnostic only.

    ``parse_query`` never turns a cue into ``ParsedQuery.time_range``: SearXNG silently skips every
    engine without time-range support (Bing, Wikipedia, Stack Overflow, GitHub, arXiv, ...) when a
    range is set, so any cue only sets ``recency_cue``. With several cues the widest range wins.
    """
    year = (today or dt.date.today()).year
    found = [name for name, pattern in _RANGE_CUES if pattern.search(text)]
    if any(int(m.group(0)) == year for m in _YEAR_RE.finditer(text)):
        found.append("year")
    time_range = max(found, key=_RANGE_WIDTH.__getitem__) if found else None
    return time_range, bool(_RECENCY_RE.search(text))


# --------------------------------------------------------------------------- keyword form

_EN_STOPWORDS_TEXT = """
a about am an and any are as at be been being best but by can could did do does doing done
for from get gets getting got had has have having he her here his how i if in into is it its
just me might my of on onto or our please shall she should so some than that the their them then
there these they this those to too us use used uses using very was way ways we were what when
where which who whom whose why will with would you your
"""
_FA_STOPWORDS_TEXT = "از به در با که این آن را و است برای چه چگونه چطور"
# "vs", "not", "no" and "without" are deliberately absent: they change what the query means.
STOPWORDS: frozenset[str] = frozenset(_EN_STOPWORDS_TEXT.split() + _FA_STOPWORDS_TEXT.split())


def _is_exact_token(term: str) -> bool:
    """Versions, error codes and identifiers: always worth keeping."""
    return any(c.isdigit() for c in term) or any(c in "_.:#+" for c in term[1:])


def keyword_form(pq: ParsedQuery, max_terms: int = 6) -> str:
    """Short deterministic keyword query for vertical engines (arXiv, PubMed, GitHub, PyPI, SE).

    Stopwords are dropped; phrases stay quoted; at most ``max_terms`` terms in original order
    (phrases last). When there are more, phrases and exact tokens (versions, error codes) are kept
    first, then the rarest words by ``wordfreq``. Escaped for SearXNG unless ``pq.raw``.
    """
    if max_terms <= 0:
        return ""
    tokens = [t for t in (_clean_token(w) for w in pq.text.split()) if any(c.isalnum() for c in t)]
    words = _dedupe(t for t in tokens if t.lower() not in STOPWORDS)
    terms = [(w, False) for w in words] + [(p, True) for p in _dedupe(pq.phrases)]
    if not terms:
        terms = [(t, False) for t in tokens[:max_terms]]
    elif len(terms) > max_terms:

        def rank(i: int) -> tuple[int, float, int]:
            term, is_phrase = terms[i]
            if is_phrase or _is_exact_token(term):
                return (0, 0.0, i)
            return (1, zipf_frequency(term.lower(), "en"), i)

        keep = sorted(sorted(range(len(terms)), key=rank)[:max_terms])
        terms = [terms[i] for i in keep]
    out = " ".join(f'"{t}"' if is_phrase else t for t, is_phrase in terms)
    return out if pq.raw else escape_searxng_syntax(out)


# --------------------------------------------------------------------------- entry point


def _scrub_into(text: str, kinds: set[str]) -> str:
    cleaned, found = scrub(text)
    kinds.update(found)
    return cleaned


def parse_query(
    query: str,
    *,
    raw: bool = False,
    language: str | None = None,
    time_range: str | None = None,
    max_chars: int = 300,
    today: dt.date | None = None,
) -> ParsedQuery:
    """Parse, scrub and cap ``query``; detect script, language and recency cues. Never raises.

    ``time_range`` comes only from the caller's argument. Time words ("today", "this week", the
    current year) and recency words ("latest") set ``recency_cue`` instead, a soft ranking boost.
    """
    original = query if isinstance(query, str) else ""
    parts = _split_operators(_normalize(original))
    kinds = set(parts.kinds)

    text = _scrub_into(" ".join(parts.words), kinds)
    phrases = _dedupe(p for p in (_scrub_into(p, kinds) for p in parts.phrases) if p)
    exclude_terms: list[str] = []
    for term in parts.exclude_terms:
        cleaned, found = scrub(term)
        kinds.update(found)
        if cleaned and not found:  # a partly scrubbed exclusion would filter the wrong thing
            exclude_terms.append(cleaned)
    parts.sites, parts.exclude_sites = _dedupe(parts.sites), _dedupe(parts.exclude_sites)
    parts.exclude_terms, parts.filetypes = _dedupe(exclude_terms), _dedupe(parts.filetypes)

    ops_len = len(" ".join(_operator_parts(parts)))
    text, phrases, truncated = _cap(text, phrases, ops_len, max_chars)
    lexical_text = " ".join(x for x in [text, *phrases] if x)

    script = detect_script(lexical_text)
    caller_language = language.strip() if isinstance(language, str) else ""
    cue_range, recency_word = time_range_cue(lexical_text, today)
    caller_range = time_range.strip().lower() if isinstance(time_range, str) else ""

    return ParsedQuery(
        original=query,
        text=text,
        lexical_text=lexical_text,
        phrases=phrases,
        sites=parts.sites,
        exclude_sites=parts.exclude_sites,
        exclude_terms=parts.exclude_terms,
        filetypes=parts.filetypes,
        script=script,
        language=caller_language or infer_language(lexical_text, script),
        time_range=caller_range if caller_range in TIME_RANGES else None,
        recency_cue=recency_word or cue_range is not None,
        scrubbed=sorted(kinds),
        truncated=truncated,
        raw=raw,
    )
