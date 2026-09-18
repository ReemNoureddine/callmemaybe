"""Choose which function to call for a natural language prompt."""

from typing import Any

from llm_sdk import Small_LLM_Model

from src.constrained_decoding import (
    generate_closed_choice,
    generate_number_value,
)
from src.models import FunctionDefinition
from src.string_candidates import build_string_candidates

BOOLEAN_CANDIDATES = ["true", "false"]


def build_seed_text(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the instruction text the model sees before it starts choosing."""
    lines = ["You are a function calling assistant. Available functions:"]

    for function in functions:
        parameter_list = []
        for name, parameter in function.parameters.items():
            parameter_list.append(f"{name}: {parameter.type}")
        signature = ", ".join(parameter_list)
        lines.append(f"- {function.name}({signature}): {function.description}")

    lines.append(f'User request: "{prompt}"')
    lines.append("Answer:")
    lines.append('{"name": "')

    return "\n".join(lines)


def find_function(
    name: str, functions: list[FunctionDefinition]
) -> FunctionDefinition:
    """Find the function definition matching `name`."""
    for function in functions:
        if function.name == name:
            return function

    raise ValueError(f"Unknown function name generated: {name!r}")


def pick_function_call(
    prompt: str,
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
    token_strings: list[str],
) -> tuple[str, dict[str, Any]]:
    """Pick a function for the given prompt and build its parameters.

    Uses constrained decoding: the model only ever chooses among tokens that
    keep the output matching a real function name and its real parameter
    types, so the result is always valid and schema-compliant.
    """
    if not functions:
        raise ValueError("No function definitions available.")

    seed_text = build_seed_text(prompt, functions)
    input_ids = model.encode(seed_text)[0].tolist()

    function_names = []
    for function in functions:
        function_names.append(function.name)

    name, input_ids = generate_closed_choice(
        model, input_ids, token_strings, function_names
    )
    function = find_function(name, functions)

    header_ids = model.encode('", "parameters": {')[0].tolist()
    input_ids = input_ids + header_ids

    parameters: dict[str, Any] = {}
    parameter_names = list(function.parameters.keys())

    for index, parameter_name in enumerate(parameter_names):
        parameter_type = function.parameters[parameter_name].type

        key_literal = f'"{parameter_name}": '
        if parameter_type == "string":
            key_literal += '"'
        input_ids = input_ids + model.encode(key_literal)[0].tolist()

        if parameter_type in ("number", "integer"):
            value_text, input_ids = generate_number_value(
                model, input_ids, token_strings
            )
            if parameter_type == "integer":
                parameters[parameter_name] = int(float(value_text))
            else:
                parameters[parameter_name] = float(value_text)
        elif parameter_type == "boolean":
            value_text, input_ids = generate_closed_choice(
                model, input_ids, token_strings, BOOLEAN_CANDIDATES
            )
            parameters[parameter_name] = value_text == "true"
        else:
            candidates = build_string_candidates(prompt, parameters)
            if candidates:
                value_text, input_ids = generate_closed_choice(
                    model, input_ids, token_strings, candidates
                )
            else:
                value_text = ""
            parameters[parameter_name] = value_text
            input_ids = input_ids + model.encode('"')[0].tolist()

        if index < len(parameter_names) - 1:
            input_ids = input_ids + model.encode(", ")[0].tolist()

    return function.name, parameters
