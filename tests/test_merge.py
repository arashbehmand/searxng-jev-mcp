"""Tests for stage E: canonical URLs, identifier keys, site keys, merge, filters, diversity."""

from __future__ import annotations

import datetime as dt
import re

import pytest

from searxng_jev_mcp.merge import (
    apply_post_filters,
    canonical_url,
    display_host,
    diversify,
    identifier_key,
    is_stackexchange,
    make_doc_id,
    merge_results,
    registrable_domain,
    site_key,
)
from searxng_jev_mcp.models import Doc, ParsedQuery, RawResult

# --- helpers ---------------------------------------------------------------------------------


def raw(
    url: str,
    *,
    title: str = "Title",
    content: str = "Snippet",
    engines: list[str] | None = None,
    positions: list[int] | None = None,
    score: float = 1.0,
    request: str = "general",
    rank: int = 1,
    category: str | None = None,
    published: dt.datetime | None = None,
) -> RawResult:
    return RawResult(
        url=url,
        title=title,
        content=content,
        engines=engines if engines is not None else ["duckduckgo"],
        positions=positions if positions is not None else [1],
        score=score,
        category=category,
        published=published,
        request=request,
        rank=rank,
    )


def doc(url: str, *, title: str = "Title", snippet: str = "Snippet", doc_id: str = "") -> Doc:
    return Doc(
        id=doc_id or make_doc_id(url, set()),
        url=url,
        canonical=canonical_url(url),
        ident=identifier_key(url),
        title=title,
        snippet=snippet,
        domain=display_host(url),
        site_key=site_key(url),
    )


def pq(
    *,
    sites: list[str] | None = None,
    exclude_sites: list[str] | None = None,
    exclude_terms: list[str] | None = None,
    phrases: list[str] | None = None,
) -> ParsedQuery:
    return ParsedQuery(
        original="q",
        text="q",
        lexical_text="q",
        phrases=phrases or [],
        sites=sites or [],
        exclude_sites=exclude_sites or [],
        exclude_terms=exclude_terms or [],
    )


# --- canonical_url ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param("http://example.com/a", "https://example.com/a", id="scheme"),
        pytest.param("https://www.example.com/a", "https://example.com/a", id="www"),
        pytest.param("https://m.example.com/a", "https://example.com/a", id="m-subdomain"),
        pytest.param("https://mobile.example.com/a", "https://example.com/a", id="mobile-subdomain"),
        pytest.param("https://amp.example.com/a", "https://example.com/a", id="amp-subdomain"),
        pytest.param(
            "https://en.m.wikipedia.org/wiki/Python", "https://en.wikipedia.org/wiki/Python", id="inner-m"
        ),
        pytest.param("https://example.com/news/story/amp", "https://example.com/news/story", id="amp-path"),
        pytest.param(
            "https://example.com/news/story/amp/", "https://example.com/news/story", id="amp-path-slash"
        ),
        pytest.param("https://example.com/story?amp=1", "https://example.com/story", id="amp-param-1"),
        pytest.param("https://example.com/story?amp", "https://example.com/story", id="amp-param-bare"),
        pytest.param(
            "https://example.com/story?outputType=amp", "https://example.com/story", id="output-type"
        ),
        pytest.param("https://example.com/story.amp.html", "https://example.com/story.html", id="amp-html"),
        pytest.param(
            "https://example-com.cdn.ampproject.org/c/s/example.com/news/story?amp_js_v=0.1",
            "https://example.com/news/story",
            id="ampproject-cache",
        ),
        pytest.param(
            "https://www.google.com/amp/s/example.com/news/story",
            "https://example.com/news/story",
            id="google-amp-viewer",
        ),
        pytest.param("https://example.com/a?utm_source=x&utm_medium=y", "https://example.com/a", id="utm"),
        pytest.param("https://example.com/a?fbclid=abc", "https://example.com/a", id="fbclid"),
        pytest.param("https://example.com/a?gclid=abc&dclid=d", "https://example.com/a", id="gclid-dclid"),
        pytest.param("https://example.com/a?msclkid=1&yclid=2", "https://example.com/a", id="msclkid-yclid"),
        pytest.param("https://example.com/a?igshid=abc", "https://example.com/a", id="igshid"),
        pytest.param("https://example.com/a?mc_cid=1&mc_eid=2", "https://example.com/a", id="mailchimp"),
        pytest.param("https://example.com/a?_hsenc=1&_hsmi=2", "https://example.com/a", id="hubspot"),
        pytest.param("https://example.com/a?ref_src=twsrc", "https://example.com/a", id="ref-src"),
        pytest.param("https://example.com/a?srsltid=AfmBOo", "https://example.com/a", id="srsltid"),
        pytest.param("https://example.com/a?UTM_Source=x", "https://example.com/a", id="utm-case"),
        pytest.param("https://example.com/a?b=2&a=1", "https://example.com/a?a=1&b=2", id="param-sort"),
        pytest.param("https://example.com/a#section-2", "https://example.com/a", id="fragment-dropped"),
        pytest.param("https://example.com/a/", "https://example.com/a", id="trailing-slash"),
        pytest.param("https://example.com", "https://example.com/", id="root"),
        pytest.param("https://example.com:443/a", "https://example.com/a", id="https-default-port"),
        pytest.param("http://example.com:80/a", "https://example.com/a", id="http-default-port"),
        pytest.param("https://EXAMPLE.com/a", "https://example.com/a", id="host-case"),
        pytest.param("https://example.com./a", "https://example.com/a", id="host-trailing-dot"),
        pytest.param("https://example.com/%7Euser", "https://example.com/~user", id="unreserved-escape"),
        pytest.param("https://example.com/a%2fb", "https://example.com/a%2Fb", id="escape-hex-case"),
        pytest.param(
            "https://fa.wikipedia.org/wiki/تهران",
            "https://fa.wikipedia.org/wiki/%D8%AA%D9%87%D8%B1%D8%A7%D9%86",
            id="iri-vs-escaped",
        ),
        pytest.param("https://user:pw@example.com/a", "https://example.com/a", id="userinfo"),
    ],
)
def test_canonical_url_equal(a: str, b: str) -> None:
    assert canonical_url(a) == canonical_url(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param("https://example.com/watch?v=abc", "https://example.com/watch?v=xyz", id="v-param"),
        pytest.param("https://example.com/item?id=1", "https://example.com/item?id=2", id="id-param"),
        pytest.param("https://example.com/?p=1", "https://example.com/?p=2", id="p-param"),
        pytest.param("https://example.com/a?ref_url=x", "https://example.com/a", id="ref-url-kept"),
        pytest.param("https://example.com/Foo", "https://example.com/foo", id="path-case"),
        pytest.param("https://example.com:8080/a", "https://example.com/a", id="custom-port"),
        pytest.param("https://example.com/app#!/users/1", "https://example.com/app", id="hashbang"),
        pytest.param("https://example.com/app#/users/1", "https://example.com/app#/users/2", id="hash-route"),
        pytest.param("https://example.com/story?amp=0", "https://example.com/story", id="amp-0-kept"),
        pytest.param("https://example.com/amp", "https://example.com/", id="bare-amp-path-kept"),
        pytest.param("https://docs.example.com/a", "https://example.com/a", id="other-subdomain"),
    ],
)
def test_canonical_url_distinct(a: str, b: str) -> None:
    assert canonical_url(a) != canonical_url(b)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "HTTPS://WWW.Example.COM:443/Path/?utm_source=x&b=2&a=1#frag",
            "example.com/Path?a=1&b=2",
        ),
        ("http://example.com", "example.com/"),
        ("https://example.com:8443/x/", "example.com:8443/x"),
        ("https://example.com/app#!/users/1", "example.com/app#!/users/1"),
        ("https://[2001:db8::1]:443/a", "[2001:db8::1]/a"),
        ("ftp://example.com/file.txt", "ftp://example.com/file.txt"),
        ("https://m.com/a", "m.com/a"),
        ("https://github.com/owner/repo/tree/main/amp", "github.com/owner/repo/tree/main/amp"),
        ("https://x.cdn.ampproject.org/c/s/", "x.cdn.ampproject.org/c/s"),
        ("https://www.example.com/a?id=7&v=x&p=2", "example.com/a?id=7&p=2&v=x"),
        ("  Not A URL  ", "not a url"),
        ("", ""),
        ("http://", "http://"),
        ("https://[::1/broken", "https://[::1/broken"),
    ],
)
def test_canonical_url_exact(url: str, expected: str) -> None:
    assert canonical_url(url) == expected


# --- identifier_key --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # arXiv
        ("https://arxiv.org/abs/2401.01234v2", "arxiv:2401.01234"),
        ("https://arxiv.org/abs/2401.01234", "arxiv:2401.01234"),
        ("https://arxiv.org/pdf/2401.01234", "arxiv:2401.01234"),
        ("https://arxiv.org/pdf/2401.01234v3.pdf", "arxiv:2401.01234"),
        ("http://export.arxiv.org/abs/2401.01234", "arxiv:2401.01234"),
        ("https://arxiv.org/html/2401.01234v1", "arxiv:2401.01234"),
        ("https://www.arxiv.org/abs/2401.01234/", "arxiv:2401.01234"),
        ("https://arxiv.org/abs/hep-th/9901001", "arxiv:hep-th/9901001"),
        ("https://arxiv.org/abs/hep-th/9901001v2", "arxiv:hep-th/9901001"),
        ("https://arxiv.org/list/cs.CL/recent", None),
        # DOI
        ("https://doi.org/10.1000/XYZ.123", "doi:10.1000/xyz.123"),
        ("http://dx.doi.org/10.1000/xyz.123", "doi:10.1000/xyz.123"),
        ("https://doi.org/10.1000%2Fxyz.123", "doi:10.1000/xyz.123"),
        ("https://doi.org/10.1000/xyz.123.", "doi:10.1000/xyz.123"),
        ("https://doi.org/10.1016/S0140-6736(20)30183-5", "doi:10.1016/s0140-6736(20)30183-5"),
        ("https://publisher.example.com/doi/10.1145/3292500.3330701", "doi:10.1145/3292500.3330701"),
        ("https://publisher.example.com/doi/abs/10.1145/3292500.3330701", "doi:10.1145/3292500.3330701"),
        ("https://publisher.example.com/doi/full/10.1002/abc.123", "doi:10.1002/abc.123"),
        ("https://publisher.example.com/doi/pdf/10.1177/0956797620", "doi:10.1177/0956797620"),
        ("https://publisher.example.com/doi/10.1002/abc.123/abstract", "doi:10.1002/abc.123"),
        ("https://link.springer.com/article/10.1007/s00134-020-06294-x", "doi:10.1007/s00134-020-06294-x"),
        ("https://doi.org/", None),
        # PubMed / PMC
        ("https://pubmed.ncbi.nlm.nih.gov/12345678/", "pmid:12345678"),
        ("https://www.ncbi.nlm.nih.gov/pubmed/12345678", "pmid:12345678"),
        ("https://europepmc.org/article/MED/12345678", "pmid:12345678"),
        ("https://europepmc.org/abstract/MED/12345678", "pmid:12345678"),
        ("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/", "pmc:1234567"),
        ("https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/", "pmc:1234567"),
        ("https://europepmc.org/article/PMC/PMC1234567", "pmc:1234567"),
        ("https://pubmed.ncbi.nlm.nih.gov/?term=cancer", None),
        # Stack Exchange
        ("https://stackoverflow.com/questions/123/how-to-x", "se:stackoverflow.com:123"),
        ("https://stackoverflow.com/q/123", "se:stackoverflow.com:123"),
        ("https://stackoverflow.com/q/123/4567", "se:stackoverflow.com:123"),
        ("https://unix.stackexchange.com/questions/42/slug", "se:unix.stackexchange.com:42"),
        ("https://superuser.com/questions/7/x", "se:superuser.com:7"),
        ("https://serverfault.com/questions/7/x", "se:serverfault.com:7"),
        ("https://askubuntu.com/questions/7/x", "se:askubuntu.com:7"),
        ("https://mathoverflow.net/questions/7/x", "se:mathoverflow.net:7"),
        ("https://stackapps.com/questions/7/x", "se:stackapps.com:7"),
        ("https://ru.stackoverflow.com/questions/7/x", "se:ru.stackoverflow.com:7"),
        ("https://pt.stackoverflow.com/questions/7", "se:pt.stackoverflow.com:7"),
        ("https://stackoverflow.com/a/456", None),
        ("https://stackoverflow.com/questions/tagged/python", None),
        ("https://stackoverflow.com/users/1/someone", None),
        # YouTube
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "yt:dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=1", "yt:dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "yt:dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?si=abc", "yt:dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "yt:dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "yt:dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "yt:dQw4w9WgXcQ"),
        ("https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ", "yt:dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/videoseries?list=PL123", None),
        ("https://www.youtube.com/@somechannel", None),
        ("https://www.youtube.com/watch?list=PL123", None),
        # GitHub
        ("https://github.com/Owner/Repo", "gh:owner/repo"),
        ("https://github.com/owner/repo/", "gh:owner/repo"),
        ("https://github.com/owner/repo.git", "gh:owner/repo"),
        ("https://www.github.com/owner/repo#readme", "gh:owner/repo"),
        ("https://github.com/owner/repo/issues/12", None),
        ("https://github.com/owner/repo/pull/3", None),
        ("https://github.com/owner/repo/blob/main/README.md", None),
        ("https://github.com/owner/repo/wiki", None),
        ("https://github.com/owner/repo/releases", None),
        ("https://github.com/owner", None),
        ("https://github.com/topics/python", None),
        ("https://github.com/orgs/example", None),
        ("https://github.com/features/actions", None),
        ("https://github.com/marketplace/actions", None),
        ("https://github.com/sponsors/someone", None),
        # Other
        ("https://example.com/page", None),
        ("not a url", None),
    ],
)
def test_identifier_key(url: str, expected: str | None) -> None:
    assert identifier_key(url) == expected


# --- hosts and site keys ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://WWW.Example.com/a", "example.com"),
        ("https://m.example.com/a", "example.com"),
        ("https://mobile.example.com/a", "example.com"),
        ("https://amp.example.com/a", "example.com"),
        ("https://en.m.wikipedia.org/wiki/X", "en.wikipedia.org"),
        ("https://docs.example.com:8080/a", "docs.example.com"),
        ("https://example-com.cdn.ampproject.org/c/s/example.com/a", "example.com"),
        ("https://192.0.2.1/a", "192.0.2.1"),
        ("nonsense", ""),
    ],
)
def test_display_host(url: str, expected: str) -> None:
    assert display_host(url) == expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("example.com", "example.com"),
        ("docs.example.com", "example.com"),
        ("a.b.example.com", "example.com"),
        ("www.bbc.co.uk", "bbc.co.uk"),
        ("news.example.co.uk", "example.co.uk"),
        ("www.example.com.au", "example.com.au"),
        ("lab.example.ac.jp", "example.ac.jp"),
        ("dept.example.ac.ir", "example.ac.ir"),
        ("someone.github.io", "someone.github.io"),
        ("docs.someone.github.io", "someone.github.io"),
        ("project.readthedocs.io", "project.readthedocs.io"),
        ("app.netlify.app", "app.netlify.app"),
        ("myblog.blogspot.com", "myblog.blogspot.com"),
        ("writer.substack.com", "writer.substack.com"),
        ("someone.medium.com", "medium.com"),
        ("github.io", "github.io"),
        ("co.uk", "co.uk"),
        ("Example.COM.", "example.com"),
        ("192.0.2.1", "192.0.2.1"),
        ("[2001:db8::1]", "[2001:db8::1]"),
        ("localhost", "localhost"),
    ],
)
def test_registrable_domain(host: str, expected: str) -> None:
    assert registrable_domain(host) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/Owner/repo", "github.com/owner"),
        ("https://github.com/owner/repo/issues/1", "github.com/owner"),
        ("https://github.com/owner", "github.com/owner"),
        ("https://gist.github.com/Owner/0123456789abcdef0123", "github.com/owner"),
        ("https://gist.github.com/0123456789abcdef0123", "github.com"),
        ("https://github.com/topics/python", "github.com"),
        ("https://github.com/", "github.com"),
        ("https://docs.python.org/3/", "python.org"),
        ("https://unix.stackexchange.com/questions/1/x", "stackexchange.com"),
        ("https://someone.github.io/post", "someone.github.io"),
        ("https://www.example.co.uk/a", "example.co.uk"),
    ],
)
def test_site_key(url: str, expected: str) -> None:
    assert site_key(url) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("stackoverflow.com", True),
        ("stackexchange.com", True),
        ("superuser.com", True),
        ("serverfault.com", True),
        ("askubuntu.com", True),
        ("mathoverflow.net", True),
        ("stackapps.com", True),
        ("example.com", False),
        ("github.com/stackoverflow", False),
    ],
)
def test_is_stackexchange(key: str, expected: bool) -> None:
    assert is_stackexchange(key) is expected


# --- merge_results ---------------------------------------------------------------------------


def test_merge_http_https_collapse_prefers_https() -> None:
    docs = merge_results([raw("http://example.com/a"), raw("https://www.example.com/a/")])
    assert len(docs) == 1
    assert docs[0].url == "https://www.example.com/a/"
    assert docs[0].domain == "example.com"
    assert docs[0].canonical == "example.com/a"


def test_merge_keeps_first_url_when_no_https() -> None:
    docs = merge_results([raw("http://example.com/a"), raw("http://www.example.com/a")])
    assert docs[0].url == "http://example.com/a"


def test_merge_keeps_longer_title_and_snippet() -> None:
    docs = merge_results(
        [
            raw("https://example.com/a", title="Short", content="A much longer snippet here"),
            raw("https://example.com/a", title="A longer title", content="Brief"),
        ]
    )
    assert docs[0].title == "A longer title"
    assert docs[0].snippet == "A much longer snippet here"


def test_merge_unions_engines_and_concatenates_positions() -> None:
    docs = merge_results(
        [
            raw("https://example.com/a", engines=["brave", "duckduckgo"], positions=[1, 3]),
            raw("https://example.com/a", engines=["mojeek"], positions=[2], request="variant1"),
        ]
    )
    assert docs[0].engines == {"brave", "duckduckgo", "mojeek"}
    assert docs[0].positions == [1, 3, 2]


def test_merge_best_rank_and_score_per_request() -> None:
    docs = merge_results(
        [
            raw("https://example.com/a", request="general", rank=5, score=0.5),
            raw("http://example.com/a", request="general", rank=2, score=0.3),
            raw("https://example.com/a", request="variant1", rank=7, score=1.5),
            raw("https://example.com/a", request="variant2", rank=0, score=0.1),
        ]
    )
    assert docs[0].request_ranks == {"general": 2, "variant1": 7}
    assert docs[0].request_scores == {"general": 0.5, "variant1": 1.5, "variant2": 0.1}


def test_merge_vertical_flag_category_and_published() -> None:
    when = dt.datetime(2024, 1, 2, tzinfo=dt.UTC)
    docs = merge_results(
        [
            raw("https://example.com/a", request="general"),
            raw("https://example.com/a", request="vertical", category="science", published=when),
            raw("https://example.com/b", request="general"),
        ]
    )
    assert docs[0].from_vertical is True
    assert docs[0].category == "science"
    assert docs[0].published == when
    assert docs[1].from_vertical is False
    assert docs[1].category is None


def test_merge_arxiv_abs_and_pdf() -> None:
    docs = merge_results(
        [
            raw("https://arxiv.org/abs/2401.01234v2", title="Paper"),
            raw("https://arxiv.org/pdf/2401.01234v3.pdf", title="[PDF] Paper title in full"),
        ]
    )
    assert len(docs) == 1
    assert docs[0].ident == "arxiv:2401.01234"
    assert docs[0].url == "https://arxiv.org/abs/2401.01234v2"
    assert docs[0].title == "[PDF] Paper title in full"


def test_merge_doi_across_doi_org_and_publisher() -> None:
    docs = merge_results(
        [
            raw("https://doi.org/10.1145/3292500.3330701"),
            raw("https://publisher.example.com/doi/abs/10.1145/3292500.3330701"),
        ]
    )
    assert len(docs) == 1
    assert docs[0].ident == "doi:10.1145/3292500.3330701"
    assert docs[0].url == "https://doi.org/10.1145/3292500.3330701"


def test_merge_upgrades_to_https_of_same_form() -> None:
    docs = merge_results(
        [
            raw("http://arxiv.org/abs/2401.01234"),
            raw("https://arxiv.org/pdf/2401.01234"),
            raw("https://arxiv.org/abs/2401.01234"),
        ]
    )
    assert docs[0].url == "https://arxiv.org/abs/2401.01234"


def test_merge_order_is_first_appearance() -> None:
    docs = merge_results(
        [
            raw("https://example.com/b"),
            raw("https://example.com/a"),
            raw("https://www.example.com/b"),
            raw("https://example.com/c"),
        ]
    )
    assert [d.canonical for d in docs] == ["example.com/b", "example.com/a", "example.com/c"]


def test_merge_does_not_collapse_similar_titles() -> None:
    docs = merge_results(
        [
            raw("https://docs.python.org/3/whatsnew/3.11.html", title="What's New In Python 3.11"),
            raw("https://docs.python.org/3/whatsnew/3.12.html", title="What's New In Python 3.12"),
        ]
    )
    assert len(docs) == 2


def test_merge_sets_keys_and_unique_ids() -> None:
    docs = merge_results([raw("https://github.com/owner/repo"), raw("https://example.com/x")])
    assert docs[0].ident == "gh:owner/repo"
    assert docs[0].site_key == "github.com/owner"
    assert docs[1].ident is None
    assert docs[1].site_key == "example.com"
    assert len({d.id for d in docs}) == 2


def test_merge_ids_do_not_depend_on_order() -> None:
    urls = ["https://example.com/a", "https://example.org/b", "https://example.net/c"]
    forward = {d.url: d.id for d in merge_results([raw(u) for u in urls])}
    backward = {d.url: d.id for d in merge_results([raw(u) for u in reversed(urls)])}
    assert forward == backward


def test_merge_empty() -> None:
    assert merge_results([]) == []


# --- make_doc_id -----------------------------------------------------------------------------

_ID_RE = re.compile(r"^[a-z][a-z0-9]{3}$")


@pytest.mark.parametrize("key", ["example.com/a", "arxiv:2401.01234", "", "ünïcode/パス"])
def test_doc_id_format_and_determinism(key: str) -> None:
    first = make_doc_id(key, set())
    assert _ID_RE.match(first)
    assert make_doc_id(key, set()) == first


def test_doc_id_added_to_taken() -> None:
    taken: set[str] = set()
    doc_id = make_doc_id("example.com/a", taken)
    assert taken == {doc_id}


def test_doc_id_collision_rehashes() -> None:
    first = make_doc_id("example.com/a", set())
    taken = {first}
    second = make_doc_id("example.com/a", taken)
    assert second != first
    assert _ID_RE.match(second)
    assert taken == {first, second}
    assert make_doc_id("example.com/a", {first}) == second  # the salted retry is deterministic


def test_doc_ids_unique_over_many_keys() -> None:
    taken: set[str] = set()
    ids = [make_doc_id(f"example.com/page/{i}", taken) for i in range(2000)]
    assert len(set(ids)) == 2000
    assert all(_ID_RE.match(i) for i in ids)


# --- apply_post_filters ----------------------------------------------------------------------

_FILTER_DOCS = [
    doc("https://www.example.com/a", title="Alpha guide"),
    doc("https://docs.example.com/b", title="Beta docs"),
    doc("https://example.org/c", title="Gamma", snippet="Pinterest board"),
    doc("https://m.example.net/pinterest-ideas", title="Delta"),
    doc("https://notexample.com/e", title="Epsilon"),
]


def _urls(docs: list[Doc]) -> list[str]:
    return [d.url for d in docs]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param(
            pq(sites=["example.com"]),
            ["https://www.example.com/a", "https://docs.example.com/b"],
            id="site-includes-subdomains",
        ),
        pytest.param(
            pq(sites=["www.example.com"]),
            ["https://www.example.com/a", "https://docs.example.com/b"],
            id="site-www",
        ),
        pytest.param(
            pq(sites=["docs.example.com"]), ["https://docs.example.com/b"], id="site-subdomain-only"
        ),
        pytest.param(
            pq(sites=["m.example.net"]), ["https://m.example.net/pinterest-ideas"], id="site-mobile-host"
        ),
        pytest.param(
            pq(sites=["example.com", "example.org"]),
            ["https://www.example.com/a", "https://docs.example.com/b", "https://example.org/c"],
            id="multiple-sites",
        ),
        pytest.param(
            pq(exclude_sites=["example.com"]),
            ["https://example.org/c", "https://m.example.net/pinterest-ideas", "https://notexample.com/e"],
            id="exclude-site",
        ),
        pytest.param(
            pq(exclude_terms=["pinterest"]),
            ["https://www.example.com/a", "https://docs.example.com/b", "https://notexample.com/e"],
            id="exclude-term-snippet-and-url",
        ),
        pytest.param(
            pq(exclude_terms=["GUIDE"]),
            [
                "https://docs.example.com/b",
                "https://example.org/c",
                "https://m.example.net/pinterest-ideas",
                "https://notexample.com/e",
            ],
            id="exclude-term-case-insensitive",
        ),
        pytest.param(pq(exclude_terms=["alp"]), _urls(_FILTER_DOCS), id="exclude-term-whole-word"),
        pytest.param(pq(phrases=["nothing matches this"]), _urls(_FILTER_DOCS), id="phrases-not-filters"),
        pytest.param(pq(), _urls(_FILTER_DOCS), id="no-filters"),
    ],
)
def test_apply_post_filters(query: ParsedQuery, expected: list[str]) -> None:
    assert _urls(apply_post_filters(_FILTER_DOCS, query)) == expected


def test_apply_post_filters_returns_new_list() -> None:
    docs = list(_FILTER_DOCS)
    out = apply_post_filters(docs, pq())
    assert out == docs
    assert out is not docs


# --- diversify -------------------------------------------------------------------------------


def _ranked(urls: list[str]) -> list[Doc]:
    return [doc(u, doc_id=f"d{i:03d}") for i, u in enumerate(urls)]


def test_diversify_caps_per_site_with_backfill() -> None:
    docs = _ranked(
        [
            "https://example.com/1",
            "https://example.com/2",
            "https://docs.example.com/3",
            "https://example.org/4",
            "https://example.net/5",
        ]
    )
    out = diversify(docs, 4)
    assert [d.url for d in out] == [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.org/4",
        "https://example.net/5",
        "https://docs.example.com/3",
    ]


def test_diversify_backfills_when_short() -> None:
    docs = _ranked([f"https://example.com/{i}" for i in range(5)] + ["https://example.org/x"])
    out = diversify(docs, 5)
    assert [d.url for d in out[:5]] == [
        "https://example.com/0",
        "https://example.com/1",
        "https://example.org/x",
        "https://example.com/2",
        "https://example.com/3",
    ]
    assert sorted(d.id for d in out) == sorted(d.id for d in docs)


def test_diversify_allows_three_for_stack_exchange() -> None:
    docs = _ranked(
        [
            "https://stackoverflow.com/questions/1/a",
            "https://stackoverflow.com/questions/2/b",
            "https://stackoverflow.com/questions/3/c",
            "https://stackoverflow.com/questions/4/d",
            "https://example.com/x",
        ]
    )
    out = diversify(docs, 4)
    assert [d.url for d in out[:4]] == [
        "https://stackoverflow.com/questions/1/a",
        "https://stackoverflow.com/questions/2/b",
        "https://stackoverflow.com/questions/3/c",
        "https://example.com/x",
    ]


def test_diversify_keys_github_by_owner() -> None:
    docs = _ranked(
        [
            "https://github.com/alpha/one",
            "https://github.com/alpha/two",
            "https://github.com/alpha/three",
            "https://github.com/beta/one",
        ]
    )
    out = diversify(docs, 3)
    assert [d.url for d in out[:3]] == [
        "https://github.com/alpha/one",
        "https://github.com/alpha/two",
        "https://github.com/beta/one",
    ]


def test_diversify_waive_keeps_order() -> None:
    docs = _ranked([f"https://example.com/{i}" for i in range(5)])
    assert diversify(docs, 3, waive=True) == docs


def test_diversify_custom_caps_and_permutation() -> None:
    docs = _ranked([f"https://example.com/{i}" for i in range(4)] + ["https://example.org/x"])
    out = diversify(docs, 3, per_site=1)
    assert [d.url for d in out] == [
        "https://example.com/0",
        "https://example.org/x",
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]


def test_diversify_beyond_k_keeps_order() -> None:
    docs = _ranked(["https://example.com/1", "https://example.org/2", "https://example.com/3"])
    assert diversify(docs, 1) == docs
    assert diversify(docs, 0) == docs
    assert diversify([], 5) == []
