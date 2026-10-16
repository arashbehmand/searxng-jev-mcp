"""Tests for stage F (lexical.py): tokenizer, background IDF, BM25F, lexical flags, RRF prior, K cut."""

from __future__ import annotations

import datetime as dt
import random

import pytest

from searxng_jev_mcp.lexical import (
    BG_IDF_CJK,
    BG_IDF_MAX,
    background_idf,
    bm25f_scores,
    compute_prior,
    is_exact_token,
    lexical_features,
    normalize,
    query_terms,
    select_candidates,
    tokenize,
    url_tokens,
    weighted_rrf,
)
from searxng_jev_mcp.models import Doc, ParsedQuery

NOW = dt.datetime(2026, 10, 8, 12, 0, tzinfo=dt.UTC)


def make_doc(
    doc_id: str,
    title: str = "",
    snippet: str = "",
    url: str | None = None,
    *,
    ranks: dict[str, int] | None = None,
    vertical: bool = False,
    published: dt.datetime | None = None,
) -> Doc:
    url = url or f"https://example.org/item/{doc_id}"
    return Doc(
        id=doc_id,
        url=url,
        canonical=url,
        ident=None,
        title=title,
        snippet=snippet,
        domain="example.org",
        site_key="example.org",
        request_ranks=dict(ranks or {}),
        from_vertical=vertical,
        published=published,
    )


def make_query(text: str, phrases: list[str] | None = None) -> ParsedQuery:
    return ParsedQuery(original=text, text=text, lexical_text=text, phrases=list(phrases or []))


# --- normalize -------------------------------------------------------------------------------------


def test_normalize_nfkc_and_casefold() -> None:
    assert normalize("ＰＹＴＨＯＮ") == "python"
    assert normalize("Straße") == "strasse"


def test_normalize_folds_arabic_letters_to_persian() -> None:
    assert normalize("كتاب") == "کتاب"
    assert normalize("علي") == "علی"
    assert normalize("مدرسة") == "مدرسه"  # teh marbuta -> heh


def test_normalize_removes_zwnj_and_zero_width() -> None:
    assert normalize("می\u200cخواهم") == "میخواهم"
    assert normalize("foo\u200bbar\u200d") == "foobar"
    assert normalize("\ufeffabc") == "abc"


def test_normalize_maps_persian_and_arabic_indic_digits() -> None:
    assert normalize("۱۲۳") == "123"
    assert normalize("٤٥") == "45"
    assert normalize("۳٫۱۲") == "3.12"


def test_normalize_strips_diacritics_and_tatweel() -> None:
    assert normalize("كَتَبَ") == "کتب"
    assert normalize("مـــن") == "من"


# --- tokenize --------------------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["3.12", "v2.0.1", "1.2.3-rc1", "2.0.0b1"])
def test_tokenize_keeps_versions_whole(version: str) -> None:
    tokens = tokenize(f"Release {version} notes")
    assert tokens == ["release", version, "notes"]


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("error[E0308]: mismatched types", "e0308"),
        ("net::ERR_SSL_PROTOCOL_ERROR in Chrome", "err_ssl_protocol_error"),
        ("failed with 0x80070005", "0x80070005"),
        ("TS2345: argument of type", "ts2345"),
        ("CVE-2024-3094 backdoor", "cve-2024-3094"),
    ],
)
def test_tokenize_keeps_error_codes(text: str, token: str) -> None:
    assert token in tokenize(text)


def test_tokenize_cpp_scope_identifier() -> None:
    assert tokenize("std::vector<int>") == ["std::vector", "std", "vector", "int"]


def test_tokenize_method_call_keeps_call_and_parts() -> None:
    assert tokenize("call foo.bar() now") == ["call", "foo.bar()", "foo", "bar", "now"]
    assert tokenize("len()") == ["len()", "len"]


def test_tokenize_dotted_name_keeps_whole_and_parts() -> None:
    assert tokenize("os.path.join") == ["os.path.join", "os", "path", "join"]


@pytest.mark.parametrize(
    ("text", "token"), [("C++", "c++"), ("C#", "c#"), (".NET", ".net"), ("Node.js", "node.js")]
)
def test_tokenize_language_names(text: str, token: str) -> None:
    assert token in tokenize(f"learn {text} today")


def test_tokenize_camel_case_keeps_original() -> None:
    assert tokenize("getUserName") == ["getusername", "get", "user", "name"]


def test_tokenize_acronym_camel_case() -> None:
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]


def test_tokenize_snake_case_keeps_original() -> None:
    assert tokenize("snake_case_name") == ["snake_case_name", "snake", "case", "name"]


def test_tokenize_camel_inside_dotted_name() -> None:
    tokens = tokenize("user.getName()")
    assert tokens[0] == "user.getname()"
    assert {"user", "getname", "get", "name"} <= set(tokens)


def test_tokenize_cjk_unigrams_and_bigrams() -> None:
    assert tokenize("数据库") == ["数", "数据", "据", "据库", "库"]


def test_tokenize_cjk_mixed_with_latin() -> None:
    assert tokenize("python编程") == ["python", "编", "编程", "程"]


def test_tokenize_persian_words_after_normalisation() -> None:
    assert tokenize("كتاب\u200cهاي خوب") == ["کتابهای", "خوب"]
    assert tokenize("نسخه ۳٫۱۲") == ["نسخه", "3.12"]


def test_tokenize_drops_pure_punctuation_and_splits_on_it() -> None:
    assert tokenize("hello, world!!! -- ... (ok)") == ["hello", "world", "ok"]
    assert tokenize("  !!! ") == []


def test_tokenize_does_not_stem() -> None:
    assert tokenize("Running runs") == ["running", "runs"]


def test_tokenize_embedded_version_in_dotted_name() -> None:
    assert "3.12" in tokenize("/usr/lib/python3.12/site-packages")


def test_url_tokens_drop_scheme_and_www_and_keep_versions() -> None:
    tokens = url_tokens("https://www.docs.python.org/3.12/whatsnew/3.12.html")
    assert "https" not in tokens
    assert "www" not in tokens
    assert {"docs", "python", "org", "whatsnew", "html", "3.12"} <= set(tokens)


def test_url_tokens_split_on_separators_and_decode() -> None:
    tokens = url_tokens("https://example.org/some_repo-name/getUserName.py?q=foo%20bar&lang=en")
    assert {
        "some",
        "repo",
        "name",
        "getusername",
        "get",
        "user",
        "py",
        "q",
        "foo",
        "bar",
        "lang",
        "en",
    } <= set(tokens)
    assert "some_repo" not in tokens


def test_url_tokens_bad_url_is_empty() -> None:
    assert url_tokens("http://[::1") == []


def test_query_terms_distinct_in_order() -> None:
    pq = make_query("rust E0308 rust mismatched E0308 types")
    assert query_terms(pq) == ["rust", "e0308", "mismatched", "types"]


# --- is_exact_token --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tok",
    [
        "3.12",
        "v2.0.1",
        "1.2.3-rc1",
        "E0308",
        "e0308",
        "ts2345",
        "0x80070005",
        "err_ssl_protocol_error",
        "cve-2024-3094",
        "x86",
    ],
)
def test_is_exact_token_positive(tok: str) -> None:
    assert is_exact_token(tok)


@pytest.mark.parametrize(
    "tok", ["python", "2026", "3", "python3", "__init__", "std::vector", "c++", "کتاب", ""]
)
def test_is_exact_token_negative(tok: str) -> None:
    assert not is_exact_token(tok)


# --- background IDF --------------------------------------------------------------------------------


def test_background_idf_rare_beats_common() -> None:
    assert background_idf("the") < background_idf("python") < background_idf("zorblaxian")


def test_background_idf_unknown_and_exact_are_max() -> None:
    assert background_idf("zorblaxian") == BG_IDF_MAX
    assert background_idf("3.12") == BG_IDF_MAX
    assert background_idf("E0308") == BG_IDF_MAX


def test_background_idf_range() -> None:
    for tok in ["the", "a", "python", "error", "version", "zorblaxian", "2026"]:
        assert 0.0 <= background_idf(tok) <= BG_IDF_MAX


def test_background_idf_persian_supported() -> None:
    common, word = background_idf("است"), background_idf("کتاب")
    assert 0.0 < common < word < BG_IDF_MAX
    assert background_idf("كتاب") == word  # Arabic letter forms normalise first


def test_background_idf_cyrillic_uses_russian() -> None:
    assert 0.0 < background_idf("и") < background_idf("книга") < BG_IDF_MAX


def test_background_idf_cjk_is_flat() -> None:
    assert background_idf("数") == BG_IDF_CJK
    assert background_idf("数据") == BG_IDF_CJK


# --- BM25F -----------------------------------------------------------------------------------------


def test_bm25f_title_match_beats_snippet_only() -> None:
    docs = [
        make_doc("a", title="borrow checker guide", snippet="notes about the compiler"),
        make_doc("b", title="compiler notes guide", snippet="about the borrow checker"),
    ]
    a, b = bm25f_scores(["borrow"], docs)
    assert a > b > 0


def test_bm25f_rare_pool_term_beats_common_pool_term() -> None:
    docs = [make_doc("rare", snippet="alpha filler")]
    docs += [make_doc(f"c{i}", snippet="common filler") for i in range(9)]
    scores = bm25f_scores(["alpha", "common"], docs, bg_blend=0.0)
    assert scores[0] > scores[1] > 0


def test_bm25f_background_rare_beats_common() -> None:
    docs = [make_doc("a", snippet="the cat"), make_doc("b", snippet="zorblaxian cat")]
    a, b = bm25f_scores(["the", "zorblaxian"], docs, bg_blend=1.0)
    assert b > a > 0


def test_bm25f_k1_saturation() -> None:
    docs = [make_doc(str(tf), snippet=" ".join(["rust"] * tf)) for tf in (1, 2, 4, 8)]
    scores = bm25f_scores(["rust"], docs, b=0.0, weights=(0.0, 1.0, 0.0))
    gains = [scores[i + 1] - scores[i] for i in range(3)]
    assert all(g > 0 for g in gains)
    assert gains[0] > gains[1] / 2 > gains[2] / 4  # per extra occurrence, returns diminish
    k1 = 1.2
    for tf, score in zip((1, 2, 4, 8), scores, strict=True):
        assert score / scores[0] == pytest.approx(tf * (k1 + 1) / (k1 + tf) / 1.0)
    assert scores[-1] < scores[0] * (k1 + 1)  # bounded by idf * (k1 + 1)


def test_bm25f_larger_k1_saturates_later() -> None:
    docs = [make_doc("1", snippet="rust"), make_doc("8", snippet=" ".join(["rust"] * 8))]
    low = bm25f_scores(["rust"], docs, k1=0.5, b=0.0)
    high = bm25f_scores(["rust"], docs, k1=3.0, b=0.0)
    assert high[1] / high[0] > low[1] / low[0]


def test_bm25f_non_negative_when_every_doc_has_the_term() -> None:
    docs = [make_doc(str(i), title="rust", snippet=f"rust doc {i}") for i in range(20)]
    scores = bm25f_scores(["rust"], docs, bg_blend=0.0)
    assert all(s > 0 for s in scores)


def test_bm25f_exact_version_beats_neighbour_version() -> None:
    docs = [
        make_doc("old", title="What's new in Python 3.11", snippet="Python python release notes"),
        make_doc("new", title="What's new in Python 3.12", snippet="Release notes"),
    ]
    old, new = bm25f_scores(query_terms(make_query("python 3.12 release notes")), docs)
    assert new > old


def test_bm25f_shorter_field_wins_with_length_normalisation() -> None:
    docs = [
        make_doc("short", snippet="rust tips"),
        make_doc("long", snippet="rust " + " ".join(f"word{i}" for i in range(30))),
    ]
    short, long_ = bm25f_scores(["rust"], docs)
    assert short > long_
    flat = bm25f_scores(["rust"], docs, b=0.0)
    assert flat[0] == pytest.approx(flat[1])


def test_bm25f_url_field_counts_less_than_snippet() -> None:
    docs = [
        make_doc("url", snippet="nothing here", url="https://example.org/tokio/guide"),
        make_doc("snip", snippet="tokio here", url="https://example.org/a/b"),
    ]
    url_only, snippet = bm25f_scores(["tokio"], docs)
    assert snippet > url_only > 0


def test_bm25f_degenerate_inputs() -> None:
    assert bm25f_scores(["rust"], []) == []
    assert bm25f_scores([], [make_doc("a", title="rust")]) == [0.0]
    assert bm25f_scores(["rust"], [make_doc("a")]) == [0.0]


def test_bm25f_duplicate_terms_count_once() -> None:
    docs = [make_doc("a", title="rust guide"), make_doc("b", snippet="go guide")]
    assert bm25f_scores(["rust", "rust"], docs) == bm25f_scores(["rust"], docs)


# --- lexical_features ------------------------------------------------------------------------------


def test_lexical_features_scores_and_ranks() -> None:
    docs = [
        make_doc("none1", title="unrelated page"),
        make_doc("weak", snippet="some rust"),
        make_doc("none2", title="another page"),
        make_doc("strong", title="Rust borrow checker", snippet="the borrow checker in rust"),
    ]
    lexical_features(make_query("rust borrow checker"), docs)
    assert [d.bm25_rank for d in docs] == [3, 2, 4, 1]
    assert docs[3].bm25 > docs[1].bm25 > 0
    assert docs[0].bm25 == 0.0
    assert docs[2].bm25 == 0.0


def test_lexical_features_coverage() -> None:
    docs = [
        make_doc("two", title="Rust borrow rules"),
        make_doc("url", title="Rust borrow rules", url="https://example.org/checker"),
        make_doc("zero", title="gardening"),
    ]
    lexical_features(make_query("rust borrow checker"), docs)
    assert docs[0].coverage == pytest.approx(2 / 3)
    assert docs[1].coverage == pytest.approx(1.0)
    assert docs[2].coverage == 0.0


def test_lexical_features_phrase_hit_normalised() -> None:
    pq = make_query('"mismatched types" rust', phrases=["Mismatched   Types"])
    docs = [
        make_doc("hit", snippet="error: MISMATCHED types found"),
        make_doc("split", snippet="types were mismatched"),
        make_doc("title", title="Mismatched\ntypes in Rust"),
    ]
    lexical_features(pq, docs)
    assert [d.phrase_hit for d in docs] == [True, False, True]


def test_lexical_features_phrase_hit_persian() -> None:
    pq = make_query("كتاب خوب", phrases=["كتاب خوب"])
    docs = [make_doc("fa", snippet="این کتاب   خوب است"), make_doc("no", snippet="کتاب بد")]
    lexical_features(pq, docs)
    assert [d.phrase_hit for d in docs] == [True, False]


def test_lexical_features_exact_hit() -> None:
    docs = [
        make_doc("hit", title="error E0308 explained"),
        make_doc("url", title="error index", url="https://doc.example.org/error_codes/E0308.html"),
        make_doc("miss", title="rust error explained"),
    ]
    lexical_features(make_query("rust error e0308"), docs)
    assert [d.exact_hit for d in docs] == [True, True, False]


def test_lexical_features_no_exact_term_means_no_exact_hit() -> None:
    docs = [make_doc("a", title="rust 2026 roadmap")]
    lexical_features(make_query("rust 2026 roadmap"), docs)
    assert not docs[0].exact_hit


def test_lexical_features_empty_query_and_empty_pool() -> None:
    lexical_features(make_query("rust"), [])
    docs = [make_doc("a", title="rust"), make_doc("b", title="go")]
    lexical_features(make_query("  "), docs)
    assert [d.bm25 for d in docs] == [0.0, 0.0]
    assert [d.bm25_rank for d in docs] == [1, 2]
    assert [d.coverage for d in docs] == [0.0, 0.0]
    assert not any(d.phrase_hit or d.exact_hit for d in docs)


# --- weighted RRF ----------------------------------------------------------------------------------


def test_weighted_rrf_math() -> None:
    scores = weighted_rrf([(["a", "b"], 1.0), (["b", "c"], 0.5)], k=20)
    assert scores["a"] == pytest.approx(1 / 21)
    assert scores["b"] == pytest.approx(1 / 22 + 0.5 / 21)
    assert scores["c"] == pytest.approx(0.5 / 22)


def test_weighted_rrf_missing_ids_and_duplicates() -> None:
    scores = weighted_rrf([(["a", "a", "b"], 1.0), ([], 1.0)], k=60)
    assert scores == {"a": pytest.approx(1 / 61), "b": pytest.approx(1 / 63)}
    assert weighted_rrf([]) == {}


# --- compute_prior ---------------------------------------------------------------------------------


def test_compute_prior_request_and_lexical_lists() -> None:
    a = make_doc("a", ranks={"general": 1})
    b = make_doc("b", ranks={"general": 2, "vertical": 1})
    b.bm25, b.coverage = 2.0, 1.0
    out = compute_prior([a, b])
    assert a.prior == pytest.approx(1 / 21)
    assert b.prior == pytest.approx(1 / 22 + 1 / 21 + 0.7 / 21)
    assert [d.id for d in out] == ["b", "a"]
    assert (b.prior_rank, a.prior_rank) == (1, 2)


def test_compute_prior_lexical_only_strong_match_climbs() -> None:
    docs = [make_doc(f"g{i}", title=f"Gardening tips part {i}", ranks={"general": i}) for i in range(1, 10)]
    strong = make_doc("exact", title="error E0308: mismatched types", ranks={"general": 10})
    docs.append(strong)
    lexical_features(make_query("E0308 mismatched types"), docs)
    out = compute_prior(docs)
    assert out[0] is strong
    assert strong.prior_rank == 1


def test_compute_prior_variant_default_weight_and_override() -> None:
    gen = make_doc("gen", ranks={"general": 1})
    var = make_doc("var", ranks={"variant1": 1})
    compute_prior([gen, var], bm25_weight=0.0)
    assert gen.prior == pytest.approx(1 / 21)
    assert var.prior == pytest.approx(0.7 / 21)
    out = compute_prior([gen, var], bm25_weight=0.0, request_weights={"variant1": 2.0})
    assert var.prior == pytest.approx(2 / 21)
    assert out[0] is var


def test_compute_prior_bm25_weight_respected() -> None:
    a = make_doc("a", ranks={"general": 1})
    b = make_doc("b", ranks={"general": 3})
    b.bm25 = 5.0
    assert compute_prior([a, b], bm25_weight=0.0)[0] is a
    assert compute_prior([a, b], bm25_weight=1.0)[0] is b


def test_compute_prior_recency_bonus() -> None:
    def build() -> list[Doc]:
        return [
            make_doc("week", ranks={"general": 1}, published=NOW - dt.timedelta(days=3)),
            make_doc("month", ranks={"general": 2}, published=NOW - dt.timedelta(days=20)),
            make_doc("old", ranks={"general": 3}, published=NOW - dt.timedelta(days=90)),
            make_doc("undated", ranks={"general": 4}),
        ]

    base = build()
    compute_prior(base, now=NOW)
    boosted = build()
    compute_prior(boosted, recency=True, now=NOW)
    unit = 1.0 / 21  # max list weight / (k + 1)
    deltas = [x.prior - y.prior for x, y in zip(boosted, base, strict=True)]
    assert deltas == pytest.approx([0.15 * unit, 0.08 * unit, 0.0, 0.0])


def test_compute_prior_recency_can_reorder_and_handles_naive_datetimes() -> None:
    stale = make_doc("stale", ranks={"general": 1})
    fresh = make_doc("fresh", ranks={"general": 2}, published=dt.datetime(2026, 10, 6, 9, 0))  # naive
    assert compute_prior([stale, fresh], recency=False, now=NOW)[0] is stale
    assert compute_prior([stale, fresh], recency=True, now=NOW)[0] is fresh


def test_compute_prior_future_dates_get_no_bonus() -> None:
    doc = make_doc("future", ranks={"general": 1}, published=NOW + dt.timedelta(days=400))
    compute_prior([doc], recency=True, now=NOW)
    assert doc.prior == pytest.approx(1 / 21)


def _prior_pool() -> list[Doc]:
    docs = [
        make_doc("a", "Rust E0308 mismatched types", "the compiler says E0308", ranks={"general": 3}),
        make_doc("b", "Rust book", "types and traits", ranks={"general": 1, "variant1": 2}),
        make_doc("c", "Mismatched types", "error E0308 in match arms", ranks={"vertical": 1}),
        make_doc("d", "Gardening", "tomatoes", ranks={"general": 2, "variant1": 1}),
        make_doc("e", "Rust forum", "types question", ranks={"vertical": 2, "general": 4}),
    ]
    lexical_features(make_query("rust E0308 mismatched types"), docs)
    return docs


def test_compute_prior_is_deterministic_and_order_independent() -> None:
    first = [(d.id, d.prior, d.prior_rank) for d in compute_prior(_prior_pool())]
    second = [(d.id, d.prior, d.prior_rank) for d in compute_prior(_prior_pool())]
    assert first == second
    shuffled = _prior_pool()
    random.Random(7).shuffle(shuffled)
    third = [(d.id, d.prior_rank) for d in compute_prior(shuffled)]
    assert third == [(i, r) for i, _, r in first]
    assert [r for _, _, r in first] == [1, 2, 3, 4, 5]


def test_compute_prior_ties_fall_back_to_input_order() -> None:
    x = make_doc("x", ranks={"general": 2, "vertical": 1})
    y = make_doc("y", ranks={"general": 1, "vertical": 2})
    z = make_doc("z")
    w = make_doc("w")
    out = compute_prior([z, x, w, y], bm25_weight=0.0)
    assert x.prior == y.prior  # same best request rank (1) too
    assert [d.id for d in out] == ["x", "y", "z", "w"]
    assert [z.prior, w.prior] == [0.0, 0.0]


def test_compute_prior_tie_prefers_better_request_rank() -> None:
    top = make_doc("top", ranks={"vertical": 1})
    later = make_doc("later", ranks={"vertical": 2})  # 2.0 / (0 + 2) == 1.0
    best = make_doc("best", ranks={"general": 1})  # 1.0 / (0 + 1) == 1.0
    out = compute_prior([later, best, top], bm25_weight=0.0, rrf_k=0, request_weights={"vertical": 2.0})
    assert later.prior == best.prior == 1.0
    assert [d.id for d in out] == ["top", "best", "later"]


def test_compute_prior_empty() -> None:
    assert compute_prior([]) == []


# --- select_candidates -----------------------------------------------------------------------------


def _ranked(n: int) -> list[Doc]:
    docs = [make_doc(f"d{i}") for i in range(n)]
    for i, d in enumerate(docs):
        d.prior, d.prior_rank = 1.0 - i / 100, i + 1
    return docs


def test_select_candidates_vertical_reservation() -> None:
    docs = _ranked(10)
    for i in (1, 8, 9):
        docs[i].from_vertical = True
    out = select_candidates(docs, 5, reserve_vertical=3, reserve_exact=0)
    assert [d.id for d in out] == ["d0", "d1", "d2", "d8", "d9"]
    assert [d.reserved for d in out] == [None, None, None, "vertical", "vertical"]
    assert [docs[3].reserved, docs[4].reserved] == [None, None]


def test_select_candidates_vertical_takes_best_prior_ones() -> None:
    docs = _ranked(10)
    for i in (6, 7, 8, 9):
        docs[i].from_vertical = True
    out = select_candidates(docs, 4, reserve_vertical=2, reserve_exact=0)
    assert [d.id for d in out] == ["d0", "d1", "d6", "d7"]


def test_select_candidates_exact_reservation_by_bm25() -> None:
    docs = _ranked(10)
    for i, bm25 in ((2, 4.0), (6, 3.0), (7, 5.0), (9, 1.0)):
        docs[i].exact_hit, docs[i].bm25 = True, bm25
    out = select_candidates(docs, 5, reserve_vertical=0, reserve_exact=2)
    assert [d.id for d in out] == ["d0", "d1", "d2", "d3", "d7"]
    assert docs[7].reserved == "exact"
    assert docs[2].reserved is None  # was already inside the cut
    assert [docs[6].reserved, docs[9].reserved] == [None, None]


def test_select_candidates_reserved_docs_in_cut_are_not_evicted() -> None:
    docs = _ranked(8)
    docs[4].from_vertical = True
    docs[3].exact_hit, docs[3].bm25 = True, 2.0
    docs[7].from_vertical = True
    out = select_candidates(docs, 5, reserve_vertical=3, reserve_exact=3)
    assert [d.id for d in out] == ["d0", "d1", "d3", "d4", "d7"]
    assert docs[7].reserved == "vertical"


def test_select_candidates_doc_both_vertical_and_exact_is_flagged_vertical() -> None:
    docs = _ranked(6)
    docs[5].from_vertical = True
    docs[5].exact_hit, docs[5].bm25 = True, 9.0
    out = select_candidates(docs, 3)
    assert [d.id for d in out] == ["d0", "d1", "d5"]
    assert docs[5].reserved == "vertical"


def test_select_candidates_k_covers_pool() -> None:
    docs = _ranked(4)
    docs[3].from_vertical = True
    out = select_candidates(docs, 4)
    assert out == docs
    assert out is not docs
    assert all(d.reserved is None for d in out)
    assert select_candidates(docs, 50) == docs


def test_select_candidates_more_reserved_than_k() -> None:
    docs = _ranked(10)
    for i in (5, 6, 7):
        docs[i].from_vertical = True
    for i in (8, 9):
        docs[i].exact_hit, docs[i].bm25 = True, 1.0
    out = select_candidates(docs, 2)
    assert len(out) == 2
    assert [d.id for d in out] == ["d5", "d6"]
    assert all(d.reserved == "vertical" for d in out)


def test_select_candidates_small_pools_and_zero_k() -> None:
    assert select_candidates([], 5) == []
    docs = _ranked(3)
    assert select_candidates(docs, 0) == []


def test_select_candidates_resets_stale_flags() -> None:
    docs = _ranked(6)
    docs[5].from_vertical = True
    select_candidates(docs, 3)
    assert docs[5].reserved == "vertical"
    docs[5].from_vertical = False
    out = select_candidates(docs, 3)
    assert [d.id for d in out] == ["d0", "d1", "d2"]
    assert docs[5].reserved is None
