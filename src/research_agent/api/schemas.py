"""
Request / response shapes for the HTTP API.

FastAPI uses these Pydantic models to:
  - validate incoming JSON (bad input -> automatic 422 error with details)
  - generate the interactive docs at /docs
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["CRISPR delivery since 2020"])
    use_cache: bool = Field(default=True, description="false = fresh OpenAlex + LLM results")


class AgentAnswer(BaseModel):
    question: str
    answer: str


class ErrorBody(BaseModel):
    detail: str
