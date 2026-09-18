"""Ground string parameters in the prompt instead of free-generating them.

Free token-by-token generation for string parameters (e.g. a `regex`
parameter) lets the model hallucinate text that never appeared in the
request. Instead, we gather a small set of candidate values that are
actually grounded in the prompt -- quoted substrings, a closed library of
regex patterns for textual-category concepts (e.g. "vowels"), and finally
the prompt's trailing unquoted word as a last-resort positional guess --
and let constrained decoding pick among only those.
"""

import re
from typing import Any

_QUOTED_SPAN = re.compile(r"'([^']*)'|\"([^\"]*)\"")
_WORD = re.compile(r"[A-Za-z]+")

REGEX_CONCEPTS: dict[str, str] = {
    "number": r"\d+", "numbers": r"\d+", "digit": r"\d+", "digits": r"\d+",
    "vowel": r"[aeiouAEIOU]", "vowels": r"[aeiouAEIOU]",
    "consonant": r"[^aeiouAEIOU\s]", "consonants": r"[^aeiouAEIOU\s]",
    "letter": r"[A-Za-z]", "letters": r"[A-Za-z]+",
    "whitespace": r"\s+", "space": r"\s+", "spaces": r"\s+",
    "punctuation": r"[^\w\s]",
    "uppercase": r"[A-Z]+", "lowercase": r"[a-z]+",
}


def extract_quoted_spans(prompt: str) -> list[str]:
    """Return every quoted substring in the prompt, in order of appearance."""
    return [
        match.group(1) if match.group(1) is not None else match.group(2)
        for match in _QUOTED_SPAN.finditer(prompt)
    ]


def extract_regex_pattern_candidates(prompt: str) -> list[str]:
    """Return built-in patterns for recognized concept words in the prompt."""
    found: list[str] = []
    for word in _WORD.findall(prompt):
        pattern = REGEX_CONCEPTS.get(word.lower())
        if pattern and pattern not in found:
            found.append(pattern)
    return found


def extract_trailing_word_candidate(prompt: str) -> list[str]:
    """Return the last standalone unquoted word in the prompt, if any."""
    unquoted = _QUOTED_SPAN.sub(" ", prompt)
    words = _WORD.findall(unquoted)
    return [words[-1]] if len(words) >= 2 else []


def build_string_candidates(
    prompt: str, already_extracted: dict[str, Any]
) -> list[str]:
    """Build grounded candidates for a string parameter from the prompt.

    Tries quoted substrings first, then known regex-concept patterns, then
    the trailing unquoted word -- each tier is only consulted if the
    previous one is empty. Values already assigned to an earlier parameter
    are excluded so the same span isn't picked twice.
    """
    used = {v for v in already_extracted.values() if isinstance(v, str)}

    quoted = dict.fromkeys(extract_quoted_spans(prompt))
    remaining = [s for s in quoted if s not in used]
    if not remaining:
        patterns = extract_regex_pattern_candidates(prompt)
        remaining = [p for p in patterns if p not in used]
    if not remaining:
        trailing = extract_trailing_word_candidate(prompt)
        remaining = [w for w in trailing if w not in used]
    return remaining
