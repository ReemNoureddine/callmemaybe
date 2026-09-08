"""Data models for the project input and output."""

from typing import Any

from pydantic import BaseModel


class FunctionParameter(BaseModel):
    """Represent the type of a function parameter."""

    type: str


class FunctionDefinition(BaseModel):
    """Represent a function that the program can choose."""

    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter


class PromptInput(BaseModel):
    """Represent one prompt from the input file."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Represent one function call written to the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]
