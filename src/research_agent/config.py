"""
All settings in one place, loaded from environment variables / .env.

Why pydantic-settings instead of os.getenv()?
  - Types are checked: RELEVANT_COUNT=abc fails at startup, not mid-run.
  - Required values (GROQ_API_KEY) give one clear error if missing.
  - Every knob can be overridden from .env without touching code,
    e.g. add  DEFAULT_RECENT_YEARS=5  to .env.

Env var names are the field names in UPPER CASE (case-insensitive).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cache_dir() -> Path:
    """
    Keep the cache OUTSIDE the project folder.

    The project lives in OneDrive; a constantly-changing database file inside a
    synced folder gets locked and re-uploaded. LOCALAPPDATA is local-only on Windows.
    """
    base = os.getenv("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / ".cache") / "research-agent"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",  # read from the folder you run the command in
        env_file_encoding="utf-8",
        extra="ignore",  # unrelated entries in .env are fine
    )

    # --- Secrets -------------------------------------------------------------
    # SecretStr hides the value when printed/logged: Settings() shows '**********'.
    groq_api_key: SecretStr
    openalex_api_key: SecretStr | None = None
    openalex_mailto: str | None = None

    # --- LLM -----------------------------------------------------------------
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.2
    # The Groq SDK retries 429/5xx itself with backoff; this sets how many times.
    llm_max_retries: int = 2
    # Extra attempts when the model returns broken/empty JSON (a random model glitch).
    llm_json_retries: int = 2

    # --- OpenAlex HTTP ---------------------------------------------------------
    openalex_base_url: str = "https://api.openalex.org"
    http_timeout_seconds: float = 30.0
    # Retries for TEMPORARY failures (timeouts, 5xx, short 429). Not for an
    # exhausted daily budget — waiting seconds won't fix that.
    http_max_retries: int = 3
    # Base wait between retries; doubles each attempt (0.5s, 1s, 2s, ...) + jitter.
    http_retry_wait_seconds: float = 0.5
    # Max requests in flight at once (OpenAlex allows ~10/s; stay polite).
    http_max_concurrency: int = 5

    # --- Response cache ----------------------------------------------------------
    # Identical OpenAlex requests are answered from disk: instant and free.
    cache_enabled: bool = True
    cache_ttl_hours: float = 24.0  # citation counts change slowly
    cache_dir: Path = Field(default_factory=_default_cache_dir)

    # --- Retrieval buckets (papers fetched per perspective) ------------------
    relevant_count: int = 10
    recent_count: int = 10
    highly_cited_count: int = 10

    # --- Recent window --------------------------------------------------------
    # Used when the user gives no time range AND the LLM gives no suggestion.
    default_recent_years: int = 3
    # Bounds for the LLM's field-pace suggestion (guards against odd answers).
    min_recent_years: int = 1
    max_recent_years: int = 10

    # --- Selection (final reading list for the summarizer) -------------------
    target_final_papers: int = 12
    # Quotas sum to target_final_papers. cited_by_top got a 3rd slot (from relevant)
    # in fix 6: once ordered by similarity, co-cited landmarks were the best picks.
    select_relevant: int = 3
    select_recent: int = 3
    select_highly_cited: int = 3
    select_cited_by_top: int = 3
    # Abstract characters sent to the summarizer per paper (keeps prompt small).
    abstract_chars_for_llm: int = 800

    # --- Semantic rerank (Gemini embeddings) -----------------------------------
    # Keyword search matches WORDS; embeddings compare MEANING. After retrieval,
    # each candidate's title+abstract is compared with the question, and papers
    # below rerank_min_similarity are dropped ("diffusion of innovations" is not
    # "diffusion models"). Skipped (with a warning) when no GEMINI_API_KEY is set.
    gemini_api_key: SecretStr | None = None
    rerank_enabled: bool = True
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768  # 768 / 1536 / 3072; 768 is plenty for reranking
    embedding_batch_size: int = 100  # texts per API request
    embedding_doc_chars: int = 1000  # abstract characters embedded per paper
    # Calibrated on 492 candidates of the 13 golden topics (2026-11-19):
    #   must-have papers 0.71-0.81 | judged relevant: median 0.73 | off-topic: median 0.65
    #   0.66 keeps all must-haves + 97% of relevant papers, removes 54% of off-topic ones.
    rerank_min_similarity: float = 0.66
    # --- Citation expansion ("snowballing") ----------------------------------
    # Landmarks often don't use the topic's keywords ("The spread of true and false
    # news online" never says "misinformation"), but the best papers on the topic
    # CITE them. Count references among the top candidates; fetch the ones cited
    # by at least expansion_min_citers of them.
    citation_expansion_enabled: bool = True
    expansion_seed_count: int = 15  # top candidates whose reference lists are read
    expansion_min_citers: int = 3  # cited by >= this many seeds -> worth fetching
    expansion_max_papers: int = 10  # at most this many new papers per question
    # Papers found ONLY by expansion must clear a stricter similarity bar than the
    # rerank threshold: whole-field "backbone" papers (LSTM, Qwen2.5, CoT for a
    # tool-use question) are co-cited too, and scored 0.66-0.68 in the fix 5 eval,
    # while expansion-found must-haves scored >= 0.69.
    expansion_min_similarity: float = 0.68

    # Embeddings of the same text never change, so they're cached long and even in
    # evals (unlike LLM answers, caching them adds no bias).
    embedding_cache_enabled: bool = True
    embedding_cache_days: float = 90.0

    # --- Europe PMC (second source, biomedical questions only) ------------------
    # OpenAlex has no abstracts for some publishers, so landmarks whose TITLE doesn't
    # name the topic are never found. Europe PMC searches PubMed abstracts + full text.
    # Slow and variable (1-30 s), so it has a hard time limit and is fail-soft.
    europepmc_enabled: bool = True
    europepmc_base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    europepmc_count: int = 25  # most-cited matches taken from Europe PMC
    europepmc_timeout_seconds: float = 20.0  # whole step (search + OpenAlex lookup)

    # --- Web app (python main.py) ---------------------------------------------
    # 127.0.0.1 = only reachable from this computer (safe default on an office network).
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Auto-restart the server when code changes (handy while developing).
    api_reload: bool = True

    # --- Tracing (Langfuse) ------------------------------------------------------
    # Every run becomes a trace: steps, LLM prompts/replies, tokens, latency,
    # OpenAlex queries. On only when both keys are set (see tracing.py).
    tracing_enabled: bool = True
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"  # EU; US: https://us.cloud.langfuse.com

    # --- Evals --------------------------------------------------------------
    # The judge grades the pipeline's output, so use a STRONGER model than the
    # one being judged. On Groq it also has its own separate token budget.
    eval_judge_model: str = "openai/gpt-oss-120b"

    # --- Tool-calling agent ---------------------------------------------------
    max_agent_steps: int = 8
    agent_default_limit: int = 5


@lru_cache
def get_settings() -> Settings:
    """
    Load settings once and reuse them.

    lru_cache makes this a cheap "singleton": the first call reads .env,
    later calls return the same object.
    """
    return Settings()
