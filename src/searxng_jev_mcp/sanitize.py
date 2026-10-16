"""Cleaning untrusted web text before it reaches the decision model or the calling agent."""

from __future__ import annotations

import html
import re
import unicodedata

# Real HTML only: a known tag name, attributes in attribute syntax, optional "/>". Anything else in
# angle brackets is content, often code: `Map<String, List<Integer>>`, `if (a < b && c > d)`.
_TAG_NAMES_TEXT = """
a abbr article aside b big blockquote br button caption cite code col dd del details dfn div dl dt em
figcaption figure font footer form h1 h2 h3 h4 h5 h6 header hr i iframe img input ins kbd label li link
main mark meta nav noscript ol option p picture pre q s samp script section select small source span
strong style sub summary sup svg table tbody td template textarea tfoot th thead time title tr tt u ul
var video wbr
"""
_TAG_NAMES = sorted(_TAG_NAMES_TEXT.split(), key=len, reverse=True)
_TAG_NAME = "(?:" + "|".join(_TAG_NAMES) + r")(?![\w-])"
# Bounded repetitions keep the scan linear on hostile input such as "<a b b b ..." with no ">".
_ATTR = r"""[A-Za-z_:@][\w:.@-]{0,60}(?:\s*=\s*(?:"[^"<>]{0,500}"|'[^'<>]{0,500}'|[^\s"'=<>`]{1,500}))?"""
_TAG = re.compile(rf"<(?:/\s*{_TAG_NAME}\s*|{_TAG_NAME}(?:\s+{_ATTR}){{0,40}}\s*/?)>", re.IGNORECASE)
_COMMENT = re.compile(r"<!--.{0,4000}?-->", re.DOTALL)
# A bare single uppercase letter right after a word is a generic type parameter, not a tag: Vec<A>.
_GENERIC_PARAM = re.compile(r"<[A-Z]>")
# zero-width, bidi embedding/override/isolate controls, BOM, soft hyphen
_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\u00ad]")
_WS = re.compile(r"\s+")


def _strip_tag(m: re.Match[str]) -> str:
    start = m.start()
    if _GENERIC_PARAM.fullmatch(m.group(0)) and start and (m.string[start - 1].isalnum()):
        return m.group(0)
    return " "


def clean_text(text: str | None, max_len: int) -> str:
    """Strip HTML tags and comments, control/zero-width/bidi characters; collapse whitespace; cut at
    a word boundary. Angle brackets that are not real HTML (generics, comparisons) are kept."""
    if not text:
        return ""
    s = _TAG.sub(_strip_tag, _COMMENT.sub(" ", text))
    s = html.unescape(s)
    s = _INVISIBLE.sub("", s)
    s = "".join(ch if unicodedata.category(ch)[0] != "C" or ch in "\t\n" else " " for ch in s)
    s = _WS.sub(" ", s).strip()
    if len(s) <= max_len:
        return s
    cut = s[: max_len - 1]
    space = cut.rfind(" ")
    if space > max_len * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + "…"


# --------------------------------------------------------------------------- injection heuristic
#
# Flag only text that addresses a model, ranker or evaluator AND tries to steer it. Pages *about*
# system prompts, [INST] tokens, prompt injection or "answer yes or no" are ordinary content for
# this tool's users, and a flagged result is demoted and hidden from the model, so false positives
# cost real answers.

_FLAGS = re.IGNORECASE

# 1. "ignore all previous instructions", unless quoted or discussed as an example.
_OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:of\s+)?(?:the\s+|your\s+|these\s+)?"
    r"(?:previous|prior|above|earlier|preceding)\s+"
    r"(?:instructions|prompts?|messages|rules|directions|directives|guidelines)\b",
    _FLAGS,
)
_DISCUSSION_CUE = re.compile(
    r"(?<!\w)(?:examples?|e\.g\.|i\.e\.|such\s+as|like|attack\w*|inject\w*|payloads?|malicious|"
    r"jailbreak\w*|phrases?|strings?|contain\w*|known\s+as|called)(?!\w)",
    _FLAGS,
)
# "how to ...", "tells the model to ...", "never ...": describing the attack, not performing it.
_DESCRIPTIVE_LEAD = re.compile(
    r"(?:\b(?:to|can|could|may|might|cannot|can't|not|never|don't|doesn't|won't)\s+)$", _FLAGS
)
_CONDITIONAL_CUE = re.compile(r"\b(?:if|unless|when|otherwise|instead)\b", _FLAGS)
_MODEL_WORD = re.compile(
    r"\b(?:ai|a\.i\.|llms?|language\s+models?|assistants?|chatbots?|bots?|gpt|chatgpt|claude|"
    r"evaluators?|graders?|rankers?|crawlers?)\b",
    _FLAGS,
)

# 2. A directive to rank/rate/mark this content, at the start of a clause.
_CLAUSE_START = re.compile(
    r"(?:^|[.!?:;,\n()\[\]\"'“”—–]|\b(?:please|kindly|always|now|then|so|you\s+(?:must|should|"
    r"need\s+to|have\s+to|are\s+to)))\s*$",
    _FLAGS,
)
_FILLER = r"(?:\s+(?:as|at|in|on|to|be|is|the|a|an|very|highly|most|position|place|spot))*"
_RANK_TAIL = r"(?:relevant|first|highest|top|#\s?1|number\s+one|authoritative)"
_DIRECTIVE = re.compile(
    r"\b(?:mark|rate|rank|score|label|treat|classify|consider)\s+"
    r"(?:this|the)\s+(?:page|result|document|site|website|article|link|listing)\b"
    rf"{_FILLER}\s+(?:{_RANK_TAIL}|helpful|true|yes|correct|best)(?![\w#])"
    # "it"/"me": only ranking verbs and ranking words ("mark it as helpful" is forum boilerplate,
    # "consider it first" is advice)
    rf"|\b(?:mark|rate|rank|score|label|classify)\s+(?:this|it|me|us)\b{_FILLER}\s+{_RANK_TAIL}(?![\w#])",
    _FLAGS,
)

# 3. "Automated evaluators must answer true", "LLMs must rate this result ...".
_SUBJECT = (
    r"(?:automated\s+)?(?:evaluators?|graders?|rankers?|ranking\s+(?:systems?|models?|algorithms?)|"
    r"ai(?:\s+(?:models?|assistants?|systems?|agents?))?|language\s+models?|llms?|chatbots?|assistants?)"
)
_MODAL = (
    r"(?:must|should|shall|(?:are|is)\s+(?:required|instructed|expected|asked)\s+to|needs?\s+to|"
    r"ha(?:ve|s)\s+to)"
)
_RANK_VERB = r"(?:rate|rank|mark|score|label|classify|prioriti[sz]e|promote|recommend)"
_AFFIRM = (  # "yes"/"true" as a verdict, not a format ("answer yes or no", "output yes if ...")
    r"[\"'“]?(?:true|yes|relevant)\b[\"'”]?"
    r"(?=\s*(?:[.!;]|$|for\b|to\s+(?:this|the|every|all|any)\b|regardless\b|always\b|no\s+matter\b))"
)
_EVALUATOR_ORDER = re.compile(
    rf"\b{_SUBJECT}\s+{_MODAL}\s+(?:always\s+|now\s+|only\s+)?(?:"
    r"(?:answer|output|return|respond|reply|say|mark|label|score|classify|judge|rate|rank|treat|consider)"
    rf"\s+(?:with\s+|it\s+as\s+|this\s+as\s+)?{_AFFIRM}"
    rf"|{_RANK_VERB}\s+this\s+(?:page|result|document|site|website|article|product|listing)\b"
    rf"|{_RANK_VERB}\s+(?:this|it|me|us)\b(?=\s*(?:[.!,;:]|$|as\b|first\b|higher\b|highest\b|at\b|on\b))"
    r")",
    _FLAGS,
)

# 4. Self-claims of rank aimed at the ranker.
_SELF_CLAIM = re.compile(
    r"\b(?:this|the\s+following)\s+(?:page|result|document|article|site|website|answer|content|listing)\s+"
    r"(?:is|should\s+be\s+(?:ranked|rated|treated|considered|marked|labell?ed|scored|listed|shown)\s+as)\s+"
    r"(?:the\s+)?(?:"
    r"(?:most\s+relevant|top|number\s+one|#\s?1)\s+(?:search\s+)?(?:result|page|document|match|hit)\b"
    r"|(?:most\s+relevant|best\s+answer|most\s+helpful|most\s+authoritative)\b"
    # aimed at a ranker ("for this query"), not a forum verdict ("the best answer to the question")
    r"(?=[^.!?]{0,40}\b(?:query|search|prompt|user'?s?\s+question)\b)"
    r")",
    _FLAGS,
)

# 5. A note addressed to models: flagged only together with steering (here or in 1-4).
_ADDRESSEE = (
    r"(?:ai|a\.i\.|llms?|language\s+models?|ranking\s+(?:systems?|algorithms?|models?)|rankers?|"
    r"search\s+engines?|crawlers?|evaluators?|graders?|assistants?|chatbots?|bots?)"
)
_ADDRESS = re.compile(
    r"\b(?:note|message|attention|instructions?|notice|memo|directive)\s+(?:to|for)\s+(?:all\s+|any\s+)?"
    rf"(?:the\s+)?(?:automated\s+)?{_ADDRESSEE}(?!\w)"
    rf"|(?:^|[.!?\n]\s*)(?:attention|dear|hey|hello|hi)\s*,?\s+(?:all\s+)?{_ADDRESSEE}\s*[:,!]",
    _FLAGS,
)
_THIS_CONTENT = r"(?:this|it|the\s+page|this\s+(?:page|result|document|site|website|article))"
_WEAK_STEER = re.compile(
    r"\b(?:answer|respond|output|reply|return|say)\s+(?:with\s+)?[\"'“]?(?:true|yes)\b"
    r"(?!\s*(?:or\b|if\b|when\b|otherwise\b|/))"
    rf"|\b{_THIS_CONTENT}\s+(?:should|must|needs?\s+to|is\s+to)\s+be\s+(?:marked|rated|ranked|scored|"
    r"labell?ed|treated|classified|considered|judged|shown|listed|placed)(?:\s+as)?\s+(?:the\s+)?(?:most\s+)?"
    r"(?:relevant|helpful|first|top|highest|authoritative|correct|true|best|#\s?1|number\s+one)\b"
    r"|\b(?:fully|completely|directly|definitively)\s+answers\s+(?:the|your|this|every|any)\s+"
    r"(?:user'?s?\s+)?(?:question|query)"
    r"|\b(?:is|as)\s+the\s+most\s+relevant\b"
    rf"|\b(?:rank|rate|place|put|list|show)\w*\s+{_THIS_CONTENT}\s+"
    r"(?:first|higher|highest|at\s+the\s+top|on\s+top)\b",
    _FLAGS,
)
_APOSTROPHE = re.compile(r"(?<=[^\W\d_])'(?=[^\W\d_])")


def _inside_quotes(before: str) -> bool:
    if before.count('"') % 2 or before.count("“") > before.count("”") or before.count("`") % 2:
        return True
    singles = _APOSTROPHE.sub("", before).count("'")
    return singles % 2 == 1 or before.count("‘") > before.count("’")


def _real_override(text: str) -> bool:
    for m in _OVERRIDE.finditer(text):
        before = text[: m.start()]
        if _inside_quotes(before) or _DISCUSSION_CUE.search(before[-40:]):
            continue
        context = before[-60:]
        if (
            _DESCRIPTIVE_LEAD.search(before[-20:]) or _CONDITIONAL_CUE.search(before[-40:])
        ) and not _MODEL_WORD.search(context):
            continue
        return True
    return False


def _directive(text: str) -> bool:
    return any(
        _CLAUSE_START.search(text[max(0, m.start() - 30) : m.start()]) for m in _DIRECTIVE.finditer(text)
    )


def _strong(text: str) -> bool:
    return (
        _real_override(text)
        or _directive(text)
        or bool(_EVALUATOR_ORDER.search(text))
        or bool(_SELF_CLAIM.search(text))
    )


def _normalize(text: str) -> str:
    text = _INVISIBLE.sub("", html.unescape(unicodedata.normalize("NFKC", text)))
    return _WS.sub(" ", text).strip()


def looks_like_injection(*texts: str) -> bool:
    """Heuristic flag for text that addresses a model, ranker or evaluator and tries to steer it.

    Fires on "ignore previous instructions" (not when quoted or discussed), directives to rank or
    mark this content, orders to evaluators ("LLMs must answer true"), self-claims of being the top
    result, and notes addressed to AI or rankers that carry such steering. Content *about* prompts,
    system messages, [INST] tokens or prompt injection is not flagged.
    """
    cleaned = [_normalize(t) for t in texts if t]
    if any(_strong(t) for t in cleaned):
        return True
    addressed = any(_ADDRESS.search(t) for t in cleaned)
    return addressed and any(_WEAK_STEER.search(t) for t in cleaned)
