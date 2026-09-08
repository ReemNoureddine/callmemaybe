*This project has been created as part of the 42 curriculum by rnouredd.*

# CallMeMaybe

## Description

CallMeMaybe is a function-calling tool: it reads natural-language prompts
(e.g. `"Greet shrek"`) and translates each one into a structured function
call (`{"name": "fn_greet", "parameters": {"name": "shrek"}}`) instead of
answering the prompt directly.

The interesting constraint is the model doing the translation: a 0.6B
parameter model (`Qwen/Qwen3-0.6B`), which is far too small to reliably
produce well-formed JSON just by prompting it and hoping. So instead of
asking the model to "please output valid JSON" and checking afterward, this
project uses **constrained decoding**: the model is only ever allowed to
choose among tokens that keep the output valid and schema-compliant, at
every single generation step. Invalid output isn't corrected after the
fact — it's never selectable in the first place.

## Instructions

Install dependencies (uses `uv`):
```
make install
```

Run against the default input files (`data/input/`), writing to
`data/output/function_calling_results.json`:
```
make run
```

Run with custom paths:
```
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Other Makefile targets: `make debug` (runs under `pdb`), `make clean`
(removes caches), `make lint` (flake8 + mypy).

**First run note:** the first time you run this, `Small_LLM_Model()`
downloads the Qwen3-0.6B weights (~1.5GB) from Hugging Face Hub and caches
them locally. If you're on a network that intercepts HTTPS (a campus
network, for instance), the download itself may fail with an SSL
certificate error — see "Challenges Faced" below for what that looks like
and how to work around it. Once the model is cached once, every later run
works fully offline; you can force offline mode explicitly with:
```
HF_HUB_OFFLINE=1 make run
```

## Algorithm Explanation

Each prompt's JSON result is built as a mix of two kinds of content:

1. **Deterministic structure**, written directly by the program: the
   braces, quotes, parameter-name keys, colons, and commas. There's no
   ambiguity at these positions given the target shape
   (`{"name": "...", "parameters": {...}}`), so nothing needs to be
   "chosen" there — the program just emits the literal text.

2. **Model-driven choices**, made through real constrained decoding, at the
   two positions that actually require understanding the prompt: which
   function to call, and what value to give each of its parameters. These
   go through the following loop, repeated once per generated token:
   - Call `get_logits_from_input_ids` to get the model's raw score for
     every one of the ~150,000 tokens in its vocabulary.
   - Reject every token that would break the required shape at this
     position. The exact rule depends on what's being generated:
     - **Function name / boolean value**: only tokens that keep the text
       generated so far a valid prefix of at least one real candidate (a
       real function name from `functions_definition.json`, or the literal
       strings `"true"`/`"false"`) are allowed. Generation stops the
       moment the text exactly matches one full candidate.
     - **Number value**: only tokens made entirely of digits, `.`, or `-`
       are allowed as continuations; a small set of separator tokens
       (comma, closing brace, whitespace) are treated as the model
       signalling "the number is finished."
     - **String value**: any token that doesn't contain a literal `"` is
       allowed as continuation; a token that is exactly `"` is treated as
       the closing quote, ending the value.
   - Among only the allowed tokens, pick whichever has the highest score
     (a restricted form of greedy decoding).

Because a token that would violate the shape is never selectable in the
first place, the result is guaranteed valid JSON and schema-compliant by
construction — not by validating and retrying after the fact.

Token text is looked up via a precomputed table (`build_token_strings`):
every token ID's text is decoded once, up front, using the SDK's public
`decode()` method, and cached in a list. This is necessary because the
masking step above needs to check potentially the entire vocabulary at
every single generation step, and calling `decode()` live each time would
be far too slow.

## Design Decisions

- **Literal structure vs. model-driven content**: rather than building a
  general-purpose JSON grammar engine that constrains every character of
  output (including punctuation), the deterministic parts of the shape are
  written directly in Python. This is simpler to implement and reason
  about, and loses nothing: there's no actual decision being made at a
  `{` or a `,`, so there's nothing for the model to meaningfully "choose"
  there anyway. Constrained decoding is reserved for the two points that
  are genuinely uncertain: the function name and each parameter's value.
- **Closed-choice generation for function names and booleans**: both are
  drawn from a small, fixed set of exact strings, so the same
  prefix-matching mechanism (`generate_closed_choice`) is reused for both,
  rather than writing two separate implementations.
- **Character-class masking for numbers and strings**: rather than trying
  to build a strict formal grammar for arbitrary numeric/string content,
  each token is checked against a simple rule (character set for numbers,
  "does not contain a quote" for strings). This keeps the implementation
  small and readable while still guaranteeing syntactic validity.
- **Precomputing the token-string lookup once per run**: the alternative
  (decoding tokens on demand during generation) would repeat the same
  ~150,000 lookups on every single generation step; doing it once up front
  and reusing a cached list keeps generation fast enough to be practical.
- **Model loaded once, passed explicitly**: `Small_LLM_Model()` and the
  token-string cache are created once in `cli.py`'s `main()` and passed
  into `pick_function_call` as arguments, rather than hidden behind
  module-level globals — this keeps the expensive one-time setup cost
  clearly separated from the per-prompt logic.
- **No private `llm_sdk` access**: only the SDK's public methods
  (`encode`, `decode`, `get_logits_from_input_ids`) are used anywhere in
  this project, as required by the subject.

## Performance Analysis

On the provided sample of 11 prompts (`data/input/function_calling_tests.json`
/ `functions_definition.json`), the first full end-to-end run produced
correct function names and correct, correctly-typed parameter values for
8 of 11 prompts (the `fn_add_numbers`, `fn_greet`, `fn_reverse_string`, and
`fn_get_square_root` cases). The remaining 3 (all calls to
`fn_substitute_string_with_regex`, the only function with three string
parameters, one of them a regex pattern) came back with truncated or
partially hallucinated string values.

Investigating the failures traced them to the string-value generator's
token cap (`max_tokens=40` in `generate_string_value`, originally set to
12): longer content like a full sentence or a regex pattern didn't fit in
the original budget, so generation was cut off mid-value before the model
had a chance to naturally emit the closing quote. The cap was raised
accordingly.

**This section should be updated with the results of a full re-run after
that fix**, along with a measured wall-clock time for the full prompt set
(to confirm it fits the subject's 5-minute budget) — both are still
pending at the time of writing, due to intermittent network access to
Hugging Face Hub on the development network (see "Challenges Faced").

## Challenges Faced

- **SSL interception on campus wifi blocking model downloads**: Hugging
  Face Hub downloads failed with
  `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed
  certificate in certificate chain`, reproducible even with plain `curl`,
  pointing to network-level HTTPS interception rather than a code issue.
  Worked around by downloading once over a different network (a personal
  hotspot) so the model gets cached locally; subsequent runs use
  `HF_HUB_OFFLINE=1` to avoid touching the network at all.
- **`mypy` resolving `llm_sdk` to the wrong directory**: `llm_sdk/` at the
  project root (the package's own build folder) shares its name with the
  actual installed package, and `mypy` was resolving imports to the empty
  outer folder instead of the real, installed one — reporting
  `Module "llm_sdk" has no attribute "Small_LLM_Model"` even though the
  code was correct. Diagnosed with `mypy --verbose` (which showed exactly
  which file it was resolving the import to) and fixed with a
  `[[tool.mypy.overrides]]` entry (`follow_imports = "skip"`) in
  `pyproject.toml`, so mypy stops trying to resolve `llm_sdk`'s internals
  entirely.
- **Getting a token's literal text without private SDK access or
  forbidden packages**: Qwen's tokenizer is a byte-level BPE tokenizer,
  where a raw vocabulary entry isn't directly human-readable text —
  correctly decoding it requires the tokenizer's internal byte-remapping
  logic. Since `transformers`/`torch` can't be imported directly in this
  project's own code, and reaching into `llm_sdk`'s private attributes is
  forbidden, token text is instead obtained via the SDK's public
  `decode()` method, precomputed once for the whole vocabulary at startup.
- **Deciding when a generated value is "done"**: the model has no
  explicit signal for "this parameter's value is complete" when its
  choices are restricted to number/string-safe tokens only. Solved by
  treating specific tokens as implicit stop markers (a literal closing
  quote for strings; comma/brace/whitespace for numbers) that end
  generation without being included in the value itself.

## Testing Strategy

Testing so far has been manual and end-to-end: running the full CLI
(`make run`) against the sample files provided in `data/input/`, then
inspecting `data/output/function_calling_results.json` prompt-by-prompt to
check that the chosen function name and parameter values actually match
what each prompt asks for. This surfaced the string-truncation issue
described above. `make lint` (flake8 + mypy, with `--disallow-untyped-defs
--check-untyped-defs`) is run after every change as a static-correctness
gate. No automated test suite (`pytest`/`unittest`) has been written yet;
this is called out in the subject as optional/not graded, but would be a
natural next step, particularly for the edge cases the subject calls out
(empty strings, large numbers, ambiguous prompts, multi-parameter
functions).

## Example Usage

```
$ make run
Wrote 11 result(s) to data/output/function_calling_results.json
```

Sample of `data/output/function_calling_results.json`:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

## Resources

- `en.subject.pdf` — the assignment brief for this project.
- [Qwen/Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
  — model card for the LLM used.
- The `llm_sdk` package provided with this project (its source and
  docstrings were read directly to understand `encode`, `decode`, and
  `get_logits_from_input_ids`).
- **AI usage**: Claude (Anthropic's Claude Code) was used throughout
  development, in an interactive, step-by-step way rather than as a
  one-shot code generator — each piece was discussed, explained, and
  reviewed before being accepted. Specifically, it was used to: design
  the overall constrained-decoding approach (the split between
  deterministic structure and model-driven choices); write the
  implementation across `src/constrained_decoding.py`,
  `src/function_caller.py`, `src/cli.py`, `src/json_parsing.py`, and
  `src/models.py`; diagnose the `mypy`/`llm_sdk` import-resolution bug and
  the campus-network SSL issue described above; and explain, line by
  line, how the generated code works, so that every part could be
  understood and defended rather than copy-pasted blindly.
