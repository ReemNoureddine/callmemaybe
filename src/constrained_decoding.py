"""Building blocks for constrained decoding.

Generate text one token at a time, but at each step only ever choose among
tokens that keep the output within a specific allowed shape (a fixed set of
strings, a number, or plain text), instead of letting the model pick freely.
"""

from llm_sdk import Small_LLM_Model

NUMBER_CHARACTERS = "0123456789.-"
STOP_MARKERS = [",", "}", " ", "\n"]
REPEAT_CHECK_LENGTH = 20


def build_token_strings(model: Small_LLM_Model) -> list[str]:
    """Decode every token ID in the model's vocabulary once.

    This lets later steps look up a token's text by its ID instead of
    calling decode() again and again during generation.
    """
    vocab_size_input_ids = model.encode(" ")[0].tolist()
    vocab_size = len(model.get_logits_from_input_ids(vocab_size_input_ids))

    token_strings = []
    for token_id in range(vocab_size):
        token_strings.append(model.decode([token_id]))

    return token_strings


def pick_best_allowed_token(
    logits: list[float], allowed_token_ids: list[int]
) -> int:
    """Return whichever of the allowed token IDs has the highest score."""
    best_token_id = allowed_token_ids[0]
    best_score = logits[best_token_id]

    for token_id in allowed_token_ids:
        if logits[token_id] > best_score:
            best_score = logits[token_id]
            best_token_id = token_id

    return best_token_id


def generate_closed_choice(
    model: Small_LLM_Model,
    input_ids: list[int],
    token_strings: list[str],
    candidates: list[str],
    max_tokens: int = 30,
) -> tuple[str, list[int]]:
    """Generate text that is forced to exactly match one of `candidates`."""
    working_ids = list(input_ids)
    generated_text = ""

    for step in range(max_tokens):
        if generated_text in candidates:
            break

        logits = model.get_logits_from_input_ids(working_ids)

        allowed_token_ids = []
        for token_id in range(len(token_strings)):
            candidate_text = generated_text + token_strings[token_id]
            for candidate in candidates:
                if candidate.startswith(candidate_text):
                    allowed_token_ids.append(token_id)
                    break

        if not allowed_token_ids:
            break

        best_token_id = pick_best_allowed_token(logits, allowed_token_ids)
        generated_text += token_strings[best_token_id]
        working_ids.append(best_token_id)

    return generated_text, working_ids


def generate_number_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    token_strings: list[str],
    max_tokens: int = 8,
) -> tuple[str, list[int]]:
    """Generate digits until the model signals the number is complete."""
    working_ids = list(input_ids)
    generated_text = ""

    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(working_ids)

        allowed_token_ids = []
        stop_token_ids = []
        for token_id in range(len(token_strings)):
            text = token_strings[token_id]
            is_number_text = text != ""
            for character in text:
                if character not in NUMBER_CHARACTERS:
                    is_number_text = False
            if text in STOP_MARKERS:
                stop_token_ids.append(token_id)
            elif is_number_text:
                allowed_token_ids.append(token_id)

        candidate_token_ids = allowed_token_ids + stop_token_ids
        if not candidate_token_ids:
            break

        best_token_id = pick_best_allowed_token(logits, candidate_token_ids)

        if best_token_id in stop_token_ids:
            break

        generated_text += token_strings[best_token_id]
        working_ids.append(best_token_id)

    if generated_text in ("", "-"):
        generated_text = "0"

    return generated_text, working_ids


def generate_string_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    token_strings: list[str],
    max_tokens: int = 40,
) -> tuple[str, list[int]]:
    """Generate plain text until the model produces a closing quote."""
    working_ids = list(input_ids)
    generated_text = ""

    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(working_ids)

        allowed_token_ids = []
        stop_token_ids = []
        for token_id in range(len(token_strings)):
            text = token_strings[token_id]
            if text == '"':
                stop_token_ids.append(token_id)
            elif "\n" in text or "\\n" in text:
                stop_token_ids.append(token_id)
            elif text != "" and '"' not in text:
                allowed_token_ids.append(token_id)

        candidate_token_ids = allowed_token_ids + stop_token_ids
        if not candidate_token_ids:
            break

        best_token_id = pick_best_allowed_token(logits, candidate_token_ids)

        if best_token_id in stop_token_ids:
            break

        generated_text += token_strings[best_token_id]
        working_ids.append(best_token_id)

        if len(generated_text) >= REPEAT_CHECK_LENGTH * 2:
            recent_chunk = generated_text[-REPEAT_CHECK_LENGTH:]
            earlier_text = generated_text[:-REPEAT_CHECK_LENGTH]
            if recent_chunk in earlier_text:
                generated_text = generated_text[:-REPEAT_CHECK_LENGTH]
                break

    return generated_text, working_ids
