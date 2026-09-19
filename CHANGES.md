# Changes

## Grounded string parameters instead of free-generating them

Free token-by-token generation for string parameters (e.g. `regex`) let the
0.6B model hallucinate text with no basis in the prompt, e.g.:

```
"Substitute the word 'cat' with 'dog' in '...'" ->
regex: "cat $1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12 $13 $14 $15 $16"
```

**Fix:** removed free generation (`generate_string_value`) entirely. Added
`src/string_candidates.py`, which builds a small set of candidates grounded
in the prompt itself — quoted substrings first, then a closed library of
regex patterns for concept words (`"vowels"` -> `[aeiouAEIOU]`), then the
trailing unquoted word as a last resort — and constrained decoding
(`generate_closed_choice`) picks only among those.

## Recovered the dropped minus sign in numeric parameters

Numeric parameters always came out unsigned, even when the prompt clearly
asked for a negative number (`"sum of -5 and 3"` -> `{"a": 3.0, "b": 5.0}`).

Two causes stacked:
1. `generate_number_value`'s character check rejected any token containing
   a space, but the tokenizer fuses the minus sign with a leading space
   (`" -"` as a single token) — so that token was excluded before it could
   even be considered.
2. Even once allowed, greedy single-token selection never picks it anyway:
   a plain digit token reliably outscores the minus token, so the model
   effectively never "chooses" to start a negative number on its own.

**Fix:** broadened the number-character check to strip a fused leading
space before validating a token (`src/constrained_decoding.py`). Since
greedy decoding still can't be trusted to pick the sign, added
`apply_sign_from_prompt()` (`src/function_caller.py`), which re-derives the
sign after generation by checking whether the prompt writes that exact
magnitude as negative — the same grounding approach as the regex fix,
applied to signs instead of whole strings.
