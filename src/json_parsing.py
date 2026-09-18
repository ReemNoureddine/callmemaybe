
import json
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionDefinition, PromptInput


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load function definitions from a JSON file."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except FileNotFoundError as error:
        raise ValueError(f"File not found: {path}") from error

    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in file: {path}") from error

    except OSError as error:
        raise ValueError(f"Could not read file: {path}") from error

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in file: {path}")

    try:
        definitions = []

        for item in data:
            definition = FunctionDefinition.model_validate(item)
            definitions.append(definition)

        return definitions
    except ValidationError as error:
        message = f"Invalid function definition in file: {path}"
        raise ValueError(message) from error


def load_prompts(path: Path) -> list[PromptInput]:
    """Load prompts from a JSON file."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except FileNotFoundError as error:
        raise ValueError(f"File not found: {path}") from error

    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in file: {path}") from error

    except OSError as error:
        raise ValueError(f"Could not read file: {path}") from error

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in file: {path}")

    try:
        prompts = []

        for item in data:
            prompt = PromptInput.model_validate(item)
            prompts.append(prompt)

        return prompts

    except ValidationError as error:
        raise ValueError(f"Invalid prompt in file: {path}") from error
