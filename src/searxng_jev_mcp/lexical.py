"""Stage F: tokenizer, BM25F with background IDF, lexical flags and the weighted-RRF prior.

BM25F knows nothing about meaning, which is why it sits next to Jev: it rewards literal tokens that a
semantic model blurs (``3.12``, ``E0308``, ``ERR_SSL_PROTOCOL_ERROR``, product names). It is (1) one
voter in the weighted-RRF prior that orders the candidates Jev sees, (2) the ranking of the no-Jev
path, and (3) the guard that reserves K-cut slots for exact-token matches. It is not the primary
ranker when Jev is up.

The pool is a few dozen snippets that were all retrieved for the query, so pool-only IDF is flat:
IDF blends non-negative Lucene IDF with a background IDF taken from ``wordfreq``.
"""

from __future__ import annotations

import datetime as dt
import functools
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from urllib.parse import unquote, urlsplit

import wordfreq

from searxng_jev_mcp.models import Doc, ParsedQuery

__all__ = [
    "BG_IDF_CJK",
    "BG_IDF_MAX",
    "background_idf",
    "bm25f_scores",
    "compute_prior",
    "is_exact_token",
    "lexical_features",
    "lexical_score",
    "normalize",
    "query_terms",
    "select_candidates",
    "tokenize",
    "url_tokens",
    "weighted_rrf",
]

# Background IDF: bg = clamp((8 - zipf) * 0.5, 0, 4). "the" (zipf 7.7) -> 0.1, "python" (3.8) -> 2.1,
# unknown words (zipf 0) and exact tokens -> 4.0. Lucene IDF over a 30-60 doc pool spans ~0..4, so the
# two halves of the blend live on the same scale.
BG_IDF_MAX = 4.0
BG_IDF_CJK = 2.0  # flat mid value for CJK grams (wordfreq needs extra deps for zh/ja)
_ZIPF_CEILING = 8.0
_ZIPF_SCALE = 0.5

# Lexical score used for the BM25F list of the prior: bm25 / pool max + these bonuses.
LEX_COVERAGE_WEIGHT = 0.3
LEX_PHRASE_WEIGHT = 0.2
LEX_EXACT_WEIGHT = 0.2

# Recency bonus, as a fraction of the largest single-list RRF contribution (max weight / (k + 1)).
RECENCY_WEEK_BONUS = 0.15
RECENCY_MONTH_BONUS = 0.08

_DEFAULT_VARIANT_WEIGHT = 0.7

# --- normalisation ---------------------------------------------------------------------------------

_DROP = [
    *range(0x200B, 0x2010),  # zero-width space, ZWNJ, ZWJ, LRM, RLM
    0x00AD,  # soft hyphen
    0x061C,  # Arabic letter mark
    0x2060,  # word joiner
    0xFEFF,  # BOM / zero-width no-break space
    *range(0x202A, 0x202F),  # bidi embeddings and overrides
    *range(0x2066, 0x206A),  # bidi isolates
    *range(0x0300, 0x0370),  # stray Latin combining marks (NFKC already composed the rest)
    0x0640,  # tatweel
    *range(0x064B, 0x0660),  # Arabic harakat, incl. hamza above/below
    0x0670,  # superscript alef
    *range(0x06D6, 0x06DD),
    *range(0x06DF, 0x06E5),
    0x06E7,
    0x06E8,
    *range(0x06EA, 0x06EE),  # Quranic annotation marks
]
_FOLD: dict[int, str | None] = {cp: None for cp in _DROP}
_FOLD.update(
    {
        ord("ي"): "ی",  # Arabic yeh -> Persian yeh
        ord("ى"): "ی",  # alef maksura -> Persian yeh
        ord("ك"): "ک",  # Arabic kaf -> keheh
        ord("ة"): "ه",  # teh marbuta -> heh (Persian spelling of Arabic loanwords)
        ord("ۀ"): "ه",  # heh with yeh above (ezafe) -> heh
        ord("أ"): "ا",
        ord("إ"): "ا",
        ord("ٱ"): "ا",
        ord("٫"): ".",  # Arabic decimal separator, so ۳٫۱۲ reads as 3.12
        ord("٬"): ",",
    }
)
_FOLD.update({0x0660 + i: str(i) for i in range(10)})  # Arabic-Indic digits
_FOLD.update({0x06F0 + i: str(i) for i in range(10)})  # Persian digits


def _prenorm(text: str) -> str:
    """Everything `normalize` does except casefolding (the tokenizer needs case for camelCase)."""
    return unicodedata.normalize("NFKC", text).translate(_FOLD)


def _fold(text: str) -> str:
    # translate again: casefold can introduce combining marks (İ -> i + U+0307)
    return text.casefold().translate(_FOLD)


def normalize(text: str) -> str:
    """NFKC, casefold, drop zero-width/bidi/ZWNJ and diacritics, fold Persian/Arabic letters and digits."""
    return _fold(_prenorm(text))


# --- tokenizer -------------------------------------------------------------------------------------

_CJK = (
    "\u1100-\u11ff\u2e80-\u2fdf\u3005\u3007\u3021-\u3029\u3040-\u309f\u30a0-\u30fa\u30fc-\u30ff"
    "\u3100-\u312f\u3130-\u318f\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7ff"
    "\uf900-\ufaff\U00020000-\U0002fa1f"
)
_W = rf"[^\W{_CJK}]"  # word character that is not CJK
_VERSION = r"v?\d+(?:\.\d+)+(?:-?(?:alpha|beta|rc|dev|pre|post|a|b)\d*(?:\.\d+)?)?"
_TOKEN_RE = re.compile(
    "|".join(
        [
            rf"(?P<cjk>[{_CJK}]+)",
            r"(?P<hex>(?<!\w)0x[0-9a-f]+(?!\w))",
            rf"(?P<ver>(?<!\w){_VERSION}(?!\w))",
            r"(?P<hcode>(?<!\w)[a-z]{2,6}-\d{2,}(?:-\d+)*(?!\w))",  # cve-2024-3094, ora-00942
            r"(?P<lang>(?<![\w+#.])(?:[cg]\+\+(?:\d\d)?|[cf]#|\.net)(?![\w+#]))",
            rf"(?P<compound>{_W}+(?:(?:::|\.){_W}+)+(?:\(\))?)",  # std::vector, os.path.join, foo.bar()
            rf"(?P<call>{_W}+\(\))",
            rf"(?P<word>{_W}+)",
        ]
    ),
    re.IGNORECASE,
)
_CAMEL_HINT_RE = re.compile(r"[a-z0-9][A-Z]|[A-Z]{2}[a-z]")
_CAMEL_SPLIT_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[^\W_A-Z]+")
_TAIL_VERSION_RE = re.compile(r"(?<=[^\W\d_])\d+(?:\.\d+)+$")  # python3.12 -> 3.12
_CJK_RE = re.compile(f"[{_CJK}]")
_ARABIC_RE = re.compile("[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")
_CYRILLIC_RE = re.compile("[\u0400-\u052f]")


def _subwords(raw: str) -> list[str]:
    """snake_case and camelCase parts of a word (casefolded), or [] when it has none."""
    pieces: list[str] = []
    for piece in raw.split("_"):
        if piece:
            pieces += _CAMEL_SPLIT_RE.findall(piece) if _CAMEL_HINT_RE.search(piece) else [piece]
    parts = [_fold(p) for p in pieces]
    return [] if parts == [_fold(raw)] else parts


def _cjk_grams(run: str) -> list[str]:
    out: list[str] = []
    for i, ch in enumerate(run):
        out.append(ch)
        if i + 1 < len(run):
            out.append(run[i : i + 2])
    return out


def tokenize(text: str) -> list[str]:
    """Normalised tokens; special tokens kept whole, compounds also emit their parts. No stemming."""
    out: list[str] = []
    for m in _TOKEN_RE.finditer(_prenorm(text)):
        kind, raw = m.lastgroup, m.group()
        if kind == "cjk":
            out += _cjk_grams(_fold(raw))
        elif kind in ("compound", "call"):
            out.append(_fold(raw))
            body = raw.removesuffix("()")
            for part in re.split(r"::|\.", body):
                out += [_fold(part), *_subwords(part)]
            if kind == "compound" and (tail := _TAIL_VERSION_RE.search(body)):
                out.append(tail.group())
        elif kind == "word":
            out += [_fold(raw), *_subwords(raw)]
        else:  # hex, ver, hcode, lang
            out.append(_fold(raw))
    return out


_URL_SEP_RE = re.compile(r"[/._\-?=&]+")
_URL_VERSION_RE = re.compile(rf"(?<!\d){_VERSION}", re.IGNORECASE)


def url_tokens(url: str) -> list[str]:
    """Tokens of a URL's host, path and query (`/._-?=&` separate), without scheme or `www`.

    Version-like substrings (`/3.12/`, `python3.12`) are also kept whole, since `.` is a separator.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
    except ValueError:
        return []
    text = unquote(f"{host} {parts.path} {parts.query}")
    versions = [_fold(m.group()) for m in _URL_VERSION_RE.finditer(_prenorm(text))]
    return [t for t in tokenize(_URL_SEP_RE.sub(" ", text)) if t != "www"] + versions


def query_terms(pq: ParsedQuery) -> list[str]:
    """Distinct tokens of the query's lexical text, in order of first appearance."""
    return list(dict.fromkeys(tokenize(pq.lexical_text)))


_EXACT_RE = re.compile(
    rf"{_VERSION}"  # 3.12, v2.0.1, 1.2.3-rc1 (any dotted number)
    r"|0x[0-9a-f]+"  # 0x80070005
    r"|[a-z]{2,6}-\d{2,}(?:-\d+)*"  # cve-2024-3094
    r"|[a-z]+\d{2,}[a-z0-9]*"  # e0308, ts2345, c2065, x86, win32
    r"|_*[a-z0-9]+(?:_+[a-z0-9]+)+_*"  # err_ssl_protocol_error (case is folded, so snake_case too)
)


def is_exact_token(tok: str) -> bool:
    """Version / error-code / constant style token. Plain words and plain integers (`2026`) are not."""
    return _EXACT_RE.fullmatch(normalize(tok)) is not None


@functools.lru_cache(maxsize=65536)
def background_idf(tok: str) -> float:
    """IDF-like weight in [0, BG_IDF_MAX] from wordfreq's zipf frequency in the token's script."""
    tok = normalize(tok)
    if not tok:
        return 0.0
    if _CJK_RE.search(tok):
        return BG_IDF_CJK
    if is_exact_token(tok):
        return BG_IDF_MAX
    lang = "fa" if _ARABIC_RE.search(tok) else "ru" if _CYRILLIC_RE.search(tok) else "en"
    zipf = wordfreq.zipf_frequency(tok, lang)
    if zipf <= 0.0:
        return BG_IDF_MAX
    return min(BG_IDF_MAX, max(0.0, (_ZIPF_CEILING - zipf) * _ZIPF_SCALE))


# --- BM25F -----------------------------------------------------------------------------------------

Fields = tuple[list[str], list[str], list[str]]  # title, snippet, URL tokens


def _fields(doc: Doc) -> Fields:
    return tokenize(doc.title), tokenize(doc.snippet), url_tokens(doc.url)


def _bm25f(
    terms: Sequence[str],
    pool: Sequence[Fields],
    k1: float,
    b: float,
    weights: Sequence[float],
    bg_blend: float,
) -> list[float]:
    terms = list(dict.fromkeys(terms))
    n_docs = len(pool)
    if not terms or n_docs == 0:
        return [0.0] * n_docs
    counts = [[Counter(field) for field in doc] for doc in pool]
    avg_len = [sum(len(doc[f]) for doc in pool) / n_docs for f in range(len(weights))]
    idf: dict[str, float] = {}
    for t in terms:
        n = sum(1 for doc in counts if any(t in field for field in doc))
        lucene = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))
        idf[t] = (1.0 - bg_blend) * lucene + bg_blend * background_idf(t)
    scores: list[float] = []
    for doc, doc_counts in zip(pool, counts, strict=True):
        norm = [
            1.0 - b + b * len(doc[f]) / avg_len[f] if avg_len[f] > 0 else 1.0 for f in range(len(weights))
        ]
        score = 0.0
        for t in terms:
            # weighted pseudo-frequency with per-field length normalisation, then one saturation
            tf = sum(w * doc_counts[f][t] / norm[f] for f, w in enumerate(weights) if doc_counts[f][t] > 0)
            if tf > 0:
                score += idf[t] * tf * (k1 + 1.0) / (k1 + tf)
        scores.append(score)
    return scores


def bm25f_scores(
    terms: list[str],
    docs: list[Doc],
    *,
    k1: float = 1.2,
    b: float = 0.3,
    weights: tuple[float, float, float] = (2.0, 1.0, 0.5),
    bg_blend: float = 0.5,
) -> list[float]:
    """BM25F over (title, snippet, URL tokens); one score per doc, in input order. Never negative."""
    return _bm25f(terms, [_fields(d) for d in docs], k1, b, weights, bg_blend)


def _squash(text: str) -> str:
    return " ".join(normalize(text).split())


def lexical_features(pq: ParsedQuery, docs: list[Doc]) -> None:
    """Set bm25, bm25_rank, coverage, phrase_hit and exact_hit on every doc."""
    terms = query_terms(pq)
    pool = [_fields(d) for d in docs]
    scores = _bm25f(terms, pool, 1.2, 0.3, (2.0, 1.0, 0.5), 0.5)
    exact_terms = [t for t in terms if is_exact_token(t)]
    phrases = [p for p in (_squash(p) for p in pq.phrases) if p]
    for rank, i in enumerate(sorted(range(len(docs)), key=lambda i: (-scores[i], i)), start=1):
        docs[i].bm25_rank = rank
    for doc, fields, score in zip(docs, pool, scores, strict=True):
        present = set(fields[0]) | set(fields[1]) | set(fields[2])
        doc.bm25 = score
        doc.coverage = sum(1 for t in terms if t in present) / len(terms) if terms else 0.0
        title, snippet = _squash(doc.title), _squash(doc.snippet)
        doc.phrase_hit = any(p in title or p in snippet for p in phrases)
        doc.exact_hit = any(t in present for t in exact_terms)


# --- prior -----------------------------------------------------------------------------------------


def weighted_rrf(rankings: list[tuple[list[str], float]], k: int = 20) -> dict[str, float]:
    """score(id) = sum over lists of weight / (k + rank), 1-based ranks; repeats within a list ignored."""
    scores: dict[str, float] = {}
    for ids, weight in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ids, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


def lexical_score(doc: Doc, max_bm25: float) -> float:
    """bm25 / pool max + 0.3·coverage + 0.2·phrase_hit + 0.2·exact_hit (range 0..1.7)."""
    norm = doc.bm25 / max_bm25 if max_bm25 > 0 else 0.0
    return (
        norm
        + LEX_COVERAGE_WEIGHT * doc.coverage
        + LEX_PHRASE_WEIGHT * doc.phrase_hit
        + LEX_EXACT_WEIGHT * doc.exact_hit
    )


def _default_request_weight(label: str) -> float:
    return _DEFAULT_VARIANT_WEIGHT if label.startswith("variant") else 1.0


def _aware(ts: dt.datetime) -> dt.datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=dt.UTC)


def compute_prior(
    docs: list[Doc],
    *,
    bm25_weight: float = 0.7,
    rrf_k: int = 20,
    request_weights: dict[str, float] | None = None,
    recency: bool = False,
    now: dt.datetime | None = None,
) -> list[Doc]:
    """Weighted RRF over one list per SearXNG request plus the lexical list; sets prior and prior_rank.

    Needs `lexical_features` to have run. Docs with no lexical evidence are left out of the lexical
    list rather than ranked by input order. Returns the docs sorted by prior (ties: best request
    rank, then input order).
    """
    if not docs:
        return []
    keys = [str(i) for i in range(len(docs))]  # index keys: robust to duplicate doc ids
    configured = request_weights or {}
    rankings: list[tuple[list[str], float]] = []
    for label in sorted({label for d in docs for label in d.request_ranks}):
        members = [i for i, d in enumerate(docs) if label in d.request_ranks]
        members.sort(key=lambda i: (docs[i].request_ranks[label], i))
        rankings.append(([keys[i] for i in members], configured.get(label, _default_request_weight(label))))
    max_bm25 = max(d.bm25 for d in docs)
    lex = [lexical_score(d, max_bm25) for d in docs]
    lex_order = sorted((i for i in range(len(docs)) if lex[i] > 0), key=lambda i: (-lex[i], i))
    rankings.append(([keys[i] for i in lex_order], bm25_weight))
    rrf = weighted_rrf(rankings, k=rrf_k)

    bonus_unit = max(w for _, w in rankings) / (rrf_k + 1)
    now_aware = _aware(now or dt.datetime.now(dt.UTC))
    for i, doc in enumerate(docs):
        doc.prior = rrf.get(keys[i], 0.0)
        if recency and doc.published is not None:
            age = now_aware - _aware(doc.published)
            if dt.timedelta(days=-1) <= age <= dt.timedelta(days=7):
                doc.prior += RECENCY_WEEK_BONUS * bonus_unit
            elif dt.timedelta(days=7) < age <= dt.timedelta(days=30):
                doc.prior += RECENCY_MONTH_BONUS * bonus_unit

    def best_rank(doc: Doc) -> float:
        return min(doc.request_ranks.values(), default=math.inf)

    order = sorted(range(len(docs)), key=lambda i: (-docs[i].prior, best_rank(docs[i]), i))
    for rank, i in enumerate(order, start=1):
        docs[i].prior_rank = rank
    return [docs[i] for i in order]


def select_candidates(
    ranked: list[Doc], K: int, *, reserve_vertical: int = 3, reserve_exact: int = 3
) -> list[Doc]:
    """Top K of `ranked` (prior order) with reserved slots for vertical and exact-token docs.

    Reserved: the best-prior `reserve_vertical` docs with from_vertical, and the `reserve_exact`
    exact_hit docs with the highest bm25. One outside the top K replaces the lowest-prior unreserved
    doc in the cut and gets `reserved` set ("vertical" wins when a doc is both). Output keeps prior order.
    """
    for doc in ranked:
        doc.reserved = None
    if K <= 0:
        return []
    if len(ranked) <= K:
        return list(ranked)
    kinds: dict[int, str] = {}
    for i in [i for i, d in enumerate(ranked) if d.from_vertical][: max(reserve_vertical, 0)]:
        kinds[i] = "vertical"
    exact = sorted((i for i, d in enumerate(ranked) if d.exact_hit), key=lambda i: (-ranked[i].bm25, i))
    for i in exact[: max(reserve_exact, 0)]:
        kinds.setdefault(i, "exact")
    chosen = set(range(K))
    for i in sorted(kinds):
        if i in chosen:
            continue
        evictable = [j for j in chosen if j not in kinds]
        if not evictable:
            break
        chosen.remove(max(evictable))
        chosen.add(i)
        ranked[i].reserved = kinds[i]
    return [ranked[i] for i in sorted(chosen)]
