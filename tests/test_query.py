"""Tests for stage A (query.py). Synthetic inputs only; no network."""

from __future__ import annotations

import base64
import datetime as dt

import pytest

from searxng_jev_mcp import query
from searxng_jev_mcp.query import (
    STOPWORDS,
    detect_script,
    escape_searxng_syntax,
    infer_language,
    keyword_form,
    parse_query,
    scrub,
    time_range_cue,
    to_searxng_q,
)

TODAY = dt.date(2026, 10, 8)
REAL_OS_USERNAME = query._os_username


def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


# Fake credentials, assembled at runtime so no literal in this file resembles a real one.
FAKE_SECRETS = {
    "openai": "sk-test-" + "0" * 24,
    "openrouter": "sk-or-v1-" + "0" * 32,
    "github": "ghp_" + "0" * 36,
    "github_pat": "github_pat_" + "0" * 30,
    "slack": "xox" + "b-" + "0" * 12,
    "aws": "AKIA" + "TEST" * 4,
    "jwt": _b64url('{"alg":"none"}') + "." + _b64url('{"sub":"test"}') + ".",
    "base64": "Xy7" * 14,  # random-looking, not hex
}


@pytest.fixture(autouse=True)
def _no_os_username(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make scrubbing independent of whoever runs the tests."""
    monkeypatch.setattr(query, "_os_username", lambda: None)


# --------------------------------------------------------------------------- operators


def test_phrase_is_extracted_and_kept_for_lexical_scoring() -> None:
    pq = parse_query('pandas "read csv" encoding')
    assert pq.phrases == ["read csv"]
    assert pq.text == "pandas encoding"
    assert pq.lexical_text == "pandas encoding read csv"


@pytest.mark.parametrize(
    ("raw", "site"),
    [
        ("site:example.com", "example.com"),
        ("site:https://www.Example.COM/", "example.com"),
        ("site:http://docs.example.com/Guide/", "docs.example.com/Guide"),
        ("SITE:example.com:8080", "example.com"),
        ("site:*.example.org", "example.org"),
    ],
)
def test_site_is_normalized(raw: str, site: str) -> None:
    pq = parse_query(f"tutorial {raw}")
    assert pq.sites == [site]
    assert pq.text == "tutorial"


def test_exclude_site_and_exclude_term() -> None:
    pq = parse_query("jaguar speed -site:Cars.example.com -car -automobile")
    assert pq.exclude_sites == ["cars.example.com"]
    assert pq.exclude_terms == ["car", "automobile"]
    assert pq.text == "jaguar speed"


def test_filetype_and_ext() -> None:
    pq = parse_query("annual report filetype:PDF ext:.xlsx")
    assert pq.filetypes == ["pdf", "xlsx"]
    assert pq.text == "annual report"


@pytest.mark.parametrize(
    "q",
    [
        "x -1",  # pure number: ambiguous, stays text
        "temperature -3.5",
        "foo-bar baz",  # hyphen inside a token
        "a - b",
        "a -> b",
        "site: example.com",  # empty operator value
        "filetype:p*f",  # invalid file type
    ],
)
def test_non_operators_stay_text(q: str) -> None:
    pq = parse_query(q)
    assert pq.text == q
    assert pq.phrases == []
    assert pq.sites == []
    assert pq.exclude_sites == []
    assert pq.exclude_terms == []
    assert pq.filetypes == []


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        # A command word makes every "-x" token a flag; flags lose their dashes, keep their place.
        ('find . -name "*.py" -type f', 'find . name type f "*.py"'),
        ("tar -xvf archive.tar.gz", "tar xvf archive.tar.gz"),
        ("ls -la /usr/local/bin", "ls la /usr/local/bin"),
        ("python -m venv", "python m venv"),
        ("pip install --upgrade", "pip install upgrade"),
        ("docker run -it --rm -detach nginx", "docker run it rm detach nginx"),
        ("grep -invert match", "grep invert match"),
        ("git push --force origin main", "git push force origin main"),
        ("kubectl get pods -n kube-system", "kubectl get pods n kube-system"),
        ("Get-ChildItem -Recurse -Filter *.log", "Get-ChildItem Recurse Filter *.log"),
        # An unmistakable flag (--long, -x) is enough even without a known command word.
        ("mytool --verbose -output report", "mytool verbose output report"),
        ("frobnicate -q -sharded", "frobnicate q sharded"),
        # No command context: flag words, short clusters and non-letter terms are flags too.
        ("what does -xvf mean", "what does xvf mean"),
        ("set -euo pipefail meaning", "set euo pipefail meaning"),
        ("predicate -maxdepth explained", "predicate maxdepth explained"),
        ("laptop -m1 review", "laptop m1 review"),
        ("option -la vs -al", "option la vs al"),
        # Numbers keep their sign.
        ("kill -9 1234", "kill -9 1234"),
    ],
)
def test_cli_flags_are_text_without_dashes(q: str, expected: str) -> None:
    pq = parse_query(q)
    assert to_searxng_q(pq) == expected
    assert pq.exclude_terms == []
    assert not [t for t in pq.lexical_text.split() if t.startswith("-") and not t[1:].isdigit()]


@pytest.mark.parametrize(
    ("q", "excluded", "expected"),
    [
        (
            "asyncio timeout -trio -anyio site:docs.python.org",
            ["trio", "anyio"],
            "asyncio timeout site:docs.python.org -trio -anyio",
        ),
        (
            '"segmentation fault" core dump gdb -windows',
            ["windows"],
            'core dump gdb "segmentation fault" -windows',
        ),
        ("jaguar speed -car -built-in", ["car", "built-in"], "jaguar speed -car -built-in"),
        ("best vpn -reddit,", ["reddit"], "best vpn -reddit"),
        (
            "قیمت دلار -بیت\u200cکوین",
            ["بیت\u200cکوین"],
            "قیمت دلار -بیت\u200cکوین",
        ),  # ZWNJ inside a Persian word
        ("قیمت طلا -دلار", ["دلار"], "قیمت طلا -دلار"),
        ('git rebase -"merge commits"', ["merge commits"], 'git rebase -"merge commits"'),  # quoted: always
        ("tar options -site:spam.example.com", [], "tar options -site:spam.example.com"),
        ("laptop deals -intitle:refurbished", ["intitle:refurbished"], "laptop deals -intitle:refurbished"),
    ],
)
def test_real_exclusions_survive(q: str, excluded: list[str], expected: str) -> None:
    pq = parse_query(q)
    assert pq.exclude_terms == excluded
    assert to_searxng_q(pq) == expected


def test_cli_context_from_a_phrase_and_flag_scrubbing() -> None:
    assert parse_query('"git rebase" -interactive').exclude_terms == []
    pq = parse_query("mytool --config=/home/alice/app/conf.toml --token=abc123")
    assert pq.text == "mytool config=conf.toml token"
    assert pq.scrubbed == ["path", "secret"]


def test_unbalanced_quote_is_text() -> None:
    pq = parse_query('"a b" "unclosed phrase')
    assert pq.phrases == ["a b"]
    assert pq.text == '"unclosed phrase'


def test_smart_quotes_make_a_phrase() -> None:
    assert parse_query("“exact words” here").phrases == ["exact words"]


def test_negated_phrase_becomes_quoted_exclusion() -> None:
    pq = parse_query('python -"monty python" snake')
    assert pq.exclude_terms == ["monty python"]
    assert to_searxng_q(pq) == 'python snake -"monty python"'


def test_combined_operators_round_trip() -> None:
    pq = parse_query(
        'pandas <2 "read csv" site:https://www.Example.com/docs/ -site:spam.example.com -legacy filetype:pdf'
    )
    assert pq.text == "pandas <2"
    assert to_searxng_q(pq) == (
        'pandas < 2 "read csv" site:example.com/docs -site:spam.example.com -legacy filetype:pdf'
    )


def test_only_operators() -> None:
    pq = parse_query("site:example.com filetype:pdf")
    assert pq.text == ""
    assert pq.lexical_text == ""
    assert to_searxng_q(pq) == "site:example.com filetype:pdf"


def test_duplicate_operators_are_deduped() -> None:
    pq = parse_query('"a b" "A B" site:example.com site:EXAMPLE.com -spam -SPAM')
    assert pq.phrases == ["a b"]
    assert pq.sites == ["example.com"]
    assert pq.exclude_terms == ["spam"]


def test_site_with_ip_is_dropped_and_reported() -> None:
    pq = parse_query("dashboard site:192.168.1.20")
    assert pq.sites == []
    assert pq.scrubbed == ["ip"]


def test_phrases_and_exclusions_are_scrubbed() -> None:
    pq = parse_query('"No such file /home/alice/app/config.yaml" -alice@example.com')
    assert pq.phrases == ["No such file config.yaml"]
    assert pq.exclude_terms == []
    assert pq.scrubbed == ["email", "path"]


# --------------------------------------------------------------------------- SearXNG escaping


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("pandas <2", "pandas < 2"),
        ("<10 results", "< 10 results"),
        (":fr", ": fr"),
        (":فارسی", ": فارسی"),
        (":zh-CN", ": zh-CN"),
        (":en_US", ": en_US"),
        (":auto", ": auto"),
        ("!wp", "! wp"),
        ("!!w", "! w"),
        ("!!", "!"),
        ("!", "!"),
        ("css !important rule", "css ! important rule"),
        ("!:fr", "! : fr"),
        ("std::vector", "std::vector"),
        ("::1", "::1"),
        (":1", ":1"),
        (":", ":"),
        ("<", "<"),
        ("<div>", "<div>"),
        ("<2.0", "<2.0"),
        ("a\t b\n c", "a b c"),
        ('"use :fr and !g here"', '"use : fr and ! g here"'),
    ],
)
def test_escape_searxng_syntax(q: str, expected: str) -> None:
    assert escape_searxng_syntax(q) == expected


@pytest.mark.parametrize("q", ["pandas <2", ":fr !!w !! :zh-CN", "!!!x <5 ::1", ":فارسی !<2", "plain text"])
def test_escape_is_idempotent(q: str) -> None:
    once = escape_searxng_syntax(q)
    assert escape_searxng_syntax(once) == once


def test_escaping_applies_inside_phrases() -> None:
    pq = parse_query('"compare :de and :fr" grammar')
    # `:fr"` keeps its closing quote, so SearXNG does not read it as a language: left alone.
    assert to_searxng_q(pq) == 'grammar "compare : de and :fr"'
    assert pq.phrases == ["compare :de and :fr"]  # stored unescaped


def test_raw_leaves_searxng_syntax_alone() -> None:
    pq = parse_query("!wp pandas <2 :fr", raw=True)
    assert pq.raw
    assert to_searxng_q(pq) == "!wp pandas <2 :fr"


# --------------------------------------------------------------------------- scrub


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("error in /Users/bob/proj/main.py", "error in main.py"),
        ("cd /home/alice/projects/app", "cd"),
        ("open ~/notes/todo.txt now", "open todo.txt now"),
        ("C:\\Users\\bob\\app\\run.exe crashed", "run.exe crashed"),
        ("C:/Users/bob/app/settings.json", "settings.json"),
        ("error at /home/alice/app/main.py:42:7 here", "error at main.py here"),
        ("permission denied /tmp/build/output", "permission denied"),
        ("cannot write /private/tmp/cache/run.lock", "cannot write run.lock"),
        ("clean /var/folders/xy/abc123/T/build", "clean"),
        ("clean /private/var/folders/xy/abc123/T/tmp.txt", "clean tmp.txt"),
        ("cat /root/.ssh/config fails", "cat fails"),
        ("open /mnt/c/Users/bob/Desktop/notes.md", "open notes.md"),
        ("ls ~bob/projects/demo", "ls"),
        ("file:///Users/bob/site/index.html blank", "index.html blank"),
        ("file:///home/alice/report.pdf viewer", "report.pdf viewer"),
        ("file:///C:/Users/bob/page.html", "page.html"),
        ("broken (/Users/bob/tool/bin) link", "broken link"),
        ("home is /Users/bob", "home is"),
        ("PATH=/usr/bin:/home/alice/bin", "PATH=/usr/bin:"),
    ],
)
def test_scrub_paths(text: str, expected: str) -> None:
    assert scrub(text) == (expected, ["path"])


@pytest.mark.parametrize(
    "text",
    [
        "difference between /dev/null and /dev/zero",
        "/etc/fstab mount options",
        "ls -la /usr/local/bin",
        "/proc/meminfo fields",
        "/opt/homebrew/bin not in PATH",
        "/var/log/syslog rotation",
        "see /var/log/app/error.log:42 for details",
        "/lib/x86_64-linux-gnu missing",
        "broken (/opt/tool/bin) link",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "file:///etc/hosts",
        "cd /tmp",
        "/usr/local/home/alice/x",
        "https://example.com/home/alice/docs",
    ],
)
def test_scrub_keeps_system_paths(text: str) -> None:
    assert scrub(text) == (text, [])


def test_system_path_survives_parse_query() -> None:
    pq = parse_query("difference between /dev/null and /dev/zero")
    assert to_searxng_q(pq) == "difference between /dev/null and /dev/zero"
    assert pq.scrubbed == []


@pytest.mark.parametrize("text", ["src/main.py", "a/b", "/usr", "and/or", "https://example.com/docs/api/v2"])
def test_scrub_keeps_relative_paths_single_segments_and_urls(text: str) -> None:
    assert scrub(text) == (text, [])


def test_scrub_os_username_only_in_paths_and_user_at_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query, "_os_username", lambda: "alice")
    assert scrub("build fails for Alice on macOS") == ("build fails for Alice on macOS", [])
    assert scrub("alice username vs alicenet") == ("alice username vs alicenet", [])
    assert scrub("ssh alice@devbox refused") == ("ssh refused", ["username"])
    assert scrub("scp x.tar alice@build-01:/srv/drop") == ("scp x.tar", ["username"])
    assert scrub("ssh bob@devbox refused") == ("ssh bob@devbox refused", [])  # not the OS user
    assert scrub("mail alice@example.com") == ("mail", ["email"])
    assert scrub("malice detection") == ("malice detection", [])


@pytest.mark.parametrize(
    ("name", "expected"),
    [("alice", "alice"), ("ab", None), ("root", None), ("runner", None), ("  ", None)],
)
def test_os_username_filter(monkeypatch: pytest.MonkeyPatch, name: str, expected: str | None) -> None:
    monkeypatch.setattr(query.getpass, "getuser", lambda: name)
    REAL_OS_USERNAME.cache_clear()
    try:
        assert REAL_OS_USERNAME() == expected
    finally:
        REAL_OS_USERNAME.cache_clear()


def test_os_username_lookup_failure_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str:
        raise OSError("no user")

    monkeypatch.setattr(query.getpass, "getuser", boom)
    REAL_OS_USERNAME.cache_clear()
    try:
        assert REAL_OS_USERNAME() is None
    finally:
        REAL_OS_USERNAME.cache_clear()


@pytest.mark.parametrize(
    ("text", "expected", "kinds"),
    [
        ("contact bob@example.com please", "contact please", ["email"]),
        ("ssh bob@10.0.0.5 refused", "ssh refused", ["email"]),
        ("git@github.com: Permission denied", "git@github.com: Permission denied", []),
        ("react@18.3.1 hydration", "react@18.3.1 hydration", []),
        ("job 550e8400-e29b-41d4-a716-446655440000 failed", "job failed", ["uuid"]),
        ("segfault at 0x7f8b3c2d1e40", "segfault at", ["hex"]),
        ("commit " + "ab12" * 10 + " broke it", "commit broke it", ["hex"]),
        ("commit a1b2c3d broke it", "commit a1b2c3d broke it", []),
        ("at 2026-10-08T14:03:22Z crash", "at crash", ["timestamp"]),
        ("[2026-10-08 14:03:22] ERROR boom", "ERROR boom", ["timestamp"]),
        ("2026-10-08 14:03:22,123 - WARNING", "- WARNING", ["timestamp"]),
        ("took until 14:03:22.123", "took until", ["timestamp"]),
        ("released 2026-10-08 in 2026", "released 2026-10-08 in 2026", []),
        ("api_key=abc123 rejected", "api_key rejected", ["secret"]),
        ('"password": "s3cretValue9"', '"password"', ["secret"]),
        ("password: how do I reset it", "password: how do I reset it", []),
        ("Authorization: Bearer abc123def456ghi", "Authorization: Bearer", ["secret"]),
        ("postgres://bob:s3cret@db.example.com/app", "postgres://db.example.com/app", ["secret"]),
        ("https://example.com/cb?token=abc123&x=1", "https://example.com/cb?token&x=1", ["secret"]),
    ],
)
def test_scrub_kinds(text: str, expected: str, kinds: list[str]) -> None:
    assert scrub(text) == (expected, kinds)


@pytest.mark.parametrize("name", sorted(FAKE_SECRETS))
def test_scrub_secret_patterns(name: str) -> None:
    secret = FAKE_SECRETS[name]
    cleaned, kinds = scrub(f"why is {secret} rejected")
    assert cleaned == "why is rejected"
    assert kinds == ["secret"]


@pytest.mark.parametrize(
    "text",
    [
        "connect 10.0.0.5 timeout",
        "connect 192.168.1.20:8080 timeout",
        "connect 100.64.0.7 timeout",
        "connect 192.0.2.10/24 timeout",
        "connect 2001:db8::1:2 timeout",
    ],
)
def test_scrub_ip_addresses(text: str) -> None:
    assert scrub(text) == ("connect timeout", ["ip"])


@pytest.mark.parametrize(
    "text",
    [
        "3.12",
        "v2.0.1",
        "1.2.3",
        "999.1.1.1",
        "E0308",
        "ERR_SSL_PROTOCOL_ERROR",
        "0x80070005",
        "std::vector",
        "foo.bar()",
        "C++",
        "C#",
        ".NET",
        "::1",
        "fe80::1",
        "2026",
        "2026-10-08",
        "sk-learn",
        "Bearer token",
        "site:example.com/docs/api/v2",
        "https://example.com/questions/12345678/How-Do-I-Merge-Two-DataFrames-In-Pandas",
        "getUserAccountSettingsForCurrentTenantById2",
        "ModuleNotFoundError: No module named 'numpy'",
    ],
)
def test_scrub_does_not_damage(text: str) -> None:
    assert scrub(text) == (text, [])


def test_scrub_reports_sorted_distinct_kinds_and_collapses_whitespace() -> None:
    text = (
        "bob@example.com   /home/alice/a/b  10.0.0.5\t\tbob@example.com  550e8400-e29b-41d4-a716-446655440000"
    )
    assert scrub(text) == ("", ["email", "ip", "path", "uuid"])
    assert parse_query(text).scrubbed == ["email", "ip", "path", "uuid"]


# --------------------------------------------------------------------------- length cap


def test_cap_truncates_at_word_boundary() -> None:
    words = [f"word{i}" for i in range(100)]
    pq = parse_query(" ".join(words))
    assert pq.truncated
    assert len(to_searxng_q(pq)) <= 300
    assert pq.text.split() == words[: len(pq.text.split())]


def test_cap_keeps_operators() -> None:
    pq = parse_query(" ".join(["lorem"] * 100) + " site:example.com filetype:pdf")
    assert pq.truncated
    assert pq.sites == ["example.com"]
    assert pq.filetypes == ["pdf"]
    assert len(to_searxng_q(pq)) <= 300


def test_cap_shortens_a_long_phrase() -> None:
    pq = parse_query('"' + " ".join(["alpha"] * 80) + '" beta', max_chars=100)
    assert pq.truncated
    assert len(to_searxng_q(pq)) <= 100
    assert len(pq.phrases) == 1
    assert set(pq.phrases[0].split()) == {"alpha"}


def test_cap_hard_cuts_text_without_spaces() -> None:
    pq = parse_query("漢" * 400)
    assert pq.truncated
    assert pq.text == "漢" * 300


def test_cap_applies_after_scrubbing() -> None:
    pq = parse_query("short " + "/home/alice/" + "x/" * 200 + "end")
    assert pq.text == "short"
    assert not pq.truncated


def test_cap_custom_and_disabled() -> None:
    assert parse_query("a b c d e", max_chars=5).text == "a b c"
    long_query = "word " * 200
    assert not parse_query(long_query, max_chars=0).truncated


# --------------------------------------------------------------------------- script / language


@pytest.mark.parametrize(
    ("text", "script", "language"),
    [
        ("چطور پایتون را نصب کنم", "arabic", "fa"),
        ("چطور pandas را نصب کنم", "arabic", "fa"),  # Persian with Latin tech terms
        ("كيف أتعلم البرمجة", "arabic", "ar"),
        ("پائتھون کیسے انسٹال کریں", "arabic", None),  # Urdu letters: not Persian
        ("如何学习编程", "cjk", "zh"),
        ("パイソンのインストール方法", "cjk", "ja"),
        ("파이썬 설치 방법", "cjk", "ko"),
        ("как установить питон", "cyrillic", "ru"),
        ("як встановити пайтон і налаштувати", "cyrillic", "uk"),
        ("how to install python", "latin", None),
        ("Ελληνικά γράμματα", "other", None),
    ],
)
def test_script_and_language(text: str, script: str, language: str | None) -> None:
    assert detect_script(text) == script
    assert infer_language(text, detect_script(text)) == language
    assert parse_query(text).language == language


def test_mostly_latin_persian_query_stays_latin() -> None:
    # 6 Persian letters vs 25 Latin: below the 40% share, so no language is forced.
    pq = parse_query("خطای ModuleNotFoundError در python")
    assert pq.script == "latin"
    assert pq.language is None


def test_no_letters_is_other() -> None:
    assert detect_script("3.12 2026 🙂") == "other"


def test_explicit_language_wins() -> None:
    assert parse_query("how to install python", language="de").language == "de"
    assert parse_query("چطور پایتون را نصب کنم", language="en").language == "en"


def test_bidi_and_zero_width_controls_are_stripped() -> None:
    pq = parse_query("\u200f\u202bسلام\u202c\u200b دنیا\ufeff")
    assert pq.text == "سلام دنیا"
    assert parse_query("می\u200cخواهم").text == "می\u200cخواهم"  # ZWNJ inside a Persian word is kept


# --------------------------------------------------------------------------- time range


@pytest.mark.parametrize(
    ("text", "time_range", "recency"),
    [
        ("python news today", "day", False),
        ("concerts tonight", "day", False),
        ("releases this week", "week", False),
        ("outages last 7 days", "week", False),
        ("security advisories past month", "month", False),
        ("prices last 30 days", "month", False),
        ("conference 2026", "year", False),
        ("conference 2025", None, False),
        ("changelog 2026-10-08", None, False),  # a date is not a standalone year
        ("version 2026.1", None, False),
        ("latest pandas version", None, True),
        ("newest gpu drivers", None, True),
        ("up to date docs", None, True),
        ("current version of numpy", None, True),
        ("USA Today crossword", None, False),
        ("today and this week", "week", False),  # widest cue wins
        ("اخبار امروز", "day", False),
        ("قیمت دلار این هفته", "week", False),
        ("آخرین نسخه پایتون", None, True),
        ("جدید\u200cترین گوشی", None, True),
        ("کنفرانس ۲۰۲۶", "year", False),
    ],
)
def test_time_range_cues(text: str, time_range: str | None, recency: bool) -> None:
    assert time_range_cue(text, TODAY) == (time_range, recency)


def test_current_year_follows_today() -> None:
    assert time_range_cue("conference 2026", dt.date(2027, 1, 1)) == (None, False)
    assert time_range_cue("conference 2027", dt.date(2027, 1, 1)) == ("year", False)


def test_caller_time_range_is_the_only_source_and_invalid_is_ignored() -> None:
    assert parse_query("news today", time_range="month", today=TODAY).time_range == "month"
    assert parse_query("news", time_range=" Week ", today=TODAY).time_range == "week"
    pq = parse_query("news today", time_range="hour", today=TODAY)
    assert pq.time_range is None
    assert pq.recency_cue
    assert parse_query("latest pandas", today=TODAY).recency_cue


@pytest.mark.parametrize(
    "q",
    [
        "python news today",
        "releases this week",
        "security advisories past month",
        "conference 2026",
        "latest pandas version",
        "اخبار امروز",
    ],
)
def test_cues_only_set_recency(q: str) -> None:
    # A time_range would silently skip every engine without time_range support (Bing, Wikipedia, ...).
    pq = parse_query(q, today=TODAY)
    assert pq.time_range is None
    assert pq.recency_cue


@pytest.mark.parametrize("q", ["conference 2025", "changelog 2026-10-08", "USA Today crossword", "pandas"])
def test_no_cue_no_recency(q: str) -> None:
    pq = parse_query(q, today=TODAY)
    assert pq.time_range is None
    assert not pq.recency_cue


# --------------------------------------------------------------------------- keyword form


def test_keyword_form_drops_stopwords() -> None:
    pq = parse_query("How do I merge two dataframes in pandas?")
    assert keyword_form(pq) == "merge two dataframes pandas"


def test_keyword_form_keeps_rarest_and_exact_terms_in_order() -> None:
    pq = parse_query("what is the best way to fix numpy 2.0 import error on python 3.12 macos ventura")
    result = keyword_form(pq)
    assert result == "numpy 2.0 python 3.12 macos ventura"
    assert len(result.split()) == 6


def test_keyword_form_keeps_phrases_quoted() -> None:
    pq = parse_query('how to "read csv" with pandas')
    assert keyword_form(pq) == 'pandas "read csv"'


def test_keyword_form_falls_back_when_everything_is_a_stopword() -> None:
    assert keyword_form(parse_query("how to do it")) == "how to do it"


def test_keyword_form_keeps_vs_and_code_tokens() -> None:
    assert keyword_form(parse_query("react vs vue")) == "react vs vue"
    assert keyword_form(parse_query("what is C++?")) == "C++"
    assert keyword_form(parse_query("why does foo.bar() return None?")) == "foo.bar() return None"


def test_keyword_form_persian_stopwords() -> None:
    assert keyword_form(parse_query("چطور پایتون را نصب کنم")) == "پایتون نصب کنم"


def test_keyword_form_is_escaped_unless_raw() -> None:
    assert keyword_form(parse_query("css !important")) == "css ! important"
    assert keyword_form(parse_query("css !important", raw=True)) == "css !important"


def test_keyword_form_max_terms() -> None:
    pq = parse_query("alpha beta gamma delta")
    assert len(keyword_form(pq, max_terms=2).split()) == 2
    assert keyword_form(pq, max_terms=0) == ""


def test_stopwords_contents() -> None:
    assert {"how", "the", "using", "را", "چطور"} <= STOPWORDS
    assert not {"vs", "not", "without"} & STOPWORDS


# --------------------------------------------------------------------------- robustness


@pytest.mark.parametrize(
    "q",
    [
        "",
        "   ",
        '"',
        '""',
        "-",
        "--",
        "site:",
        "-site:",
        "!!",
        ":",
        "<",
        "🙂🙂",
        "\u202e\u200f",
        "\u200c",
        "\x00\x07 x",
        "a" * 5000,
        '"' * 7,
    ],
)
def test_odd_input_never_raises(q: str) -> None:
    pq = parse_query(q)
    assert pq.original == q
    assert isinstance(to_searxng_q(pq), str)
    assert isinstance(keyword_form(pq), str)
    assert len(to_searxng_q(pq)) <= 310


def test_original_is_kept_verbatim() -> None:
    q = "  /Users/bob/x.py   site:example.com  "
    pq = parse_query(q)
    assert pq.original == q
    assert pq.text == "x.py"


@pytest.mark.parametrize(
    ("q", "expected_q", "excluded"),
    [
        ("how to make bread -yeast", "how to make bread -yeast", ["yeast"]),  # "make" is not a command here
        (
            "sort a list in python -numpy",
            "sort a list in python -numpy",
            ["numpy"],
        ),  # weak context: long word
        ("tar -xvf archive.tar.gz", "tar xvf archive.tar.gz", []),  # command right before a flag cluster
        ("how to use grep -rn recursively", "how to use grep rn recursively", []),
        ("ffmpeg -i in.mp4 -vcodec libx264", "ffmpeg i in.mp4 vcodec libx264", []),  # -i: strong context
    ],
)
def test_command_context_needs_evidence(q: str, expected_q: str, excluded: list[str]) -> None:
    pq = parse_query(q)
    assert to_searxng_q(pq) == expected_q
    assert pq.exclude_terms == excluded
