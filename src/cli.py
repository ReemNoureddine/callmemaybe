"""Command-line interface for CallMeMaybe."""

import argparse
import json
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.constrained_decoding import build_token_strings
from src.function_caller import pick_function_call
from src.json_parsing import load_function_definitions, load_prompts
from src.models import FunctionCallResult

DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for the application."""
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into structured"
                    " function calls."
    )

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help="Path to the JSON file listing the available functions.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the JSON file listing the prompts to process.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the JSON file the results will be written to.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the CallMeMaybe command-line application."""
    args = parse_args()

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompts(args.input)
    except ValueError as error:
        print(f"Error: {error}")
        return

    try:
        model = Small_LLM_Model()
        token_strings = build_token_strings(model)
    except Exception as error:
        print(f"Error: could not load the language model: {error}")
        return

    results = []
    for prompt_input in prompts:
        try:
            name, parameters = pick_function_call(
                prompt_input.prompt, functions, model, token_strings
            )
        except Exception as error:
            print(
                f"Error: could not process prompt {prompt_input.prompt!r}: "
                f"{error}"
            )
            continue
        result = FunctionCallResult(
            prompt=prompt_input.prompt, name=name, parameters=parameters
        )
        results.append(result)

    output_data = []
    for result in results:
        output_data.append(result.model_dump())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

    print(f"Wrote {len(results)} result(s) to {args.output}")
