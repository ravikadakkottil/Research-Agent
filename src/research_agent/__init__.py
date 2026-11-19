"""
research_agent — find relevant, influential and recent papers and summarize them.

Package map (read in this order):
    config.py              settings from .env (one place for every knob)
    models.py              data shapes: Paper, QueryPlan, ResearchSummary, ...
    llm.py                 Groq client + "ask for JSON, validate with Pydantic"
    prompts.py             all system prompts
    cache.py               on-disk response cache (SQLite)
    sources/openalex.py    the ONLY code that talks to OpenAlex (async, retries, cache)
    planning.py            LLM #1: question -> QueryPlan
    retrieval/recency.py   decide the "recent" window (user > field pace > default)
    retrieval/buckets.py   relevant / recent / highly_cited searches
    retrieval/selection.py dedupe + balanced selection
    synthesis.py           LLM #2: papers -> ResearchSummary
    pipeline.py            glue: runs the steps in order (no print/input)
    agent/                 tool-calling agent (LLM chooses tools)
    cli.py                 terminal interface (the only place with input/print)
    api/                   FastAPI web app + streaming + web page (run: python main.py)
"""

__version__ = "0.4.0"
