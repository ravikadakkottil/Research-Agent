# Research Agent — learning GenAI step by step

Ask about any research topic and get a balanced reading list (relevant, influential and
recent papers from OpenAlex) plus a grounded summary.

- `src/research_agent/` — the app, structured as a proper Python package (Step 2)
- `lessons/` — the original learning scripts, kept unchanged for reference

---

## Setup (first time)

1. Get keys:
   - Groq: [https://console.groq.com](https://console.groq.com)
   - OpenAlex (free, avoids 429 errors): [https://openalex.org/settings/api](https://openalex.org/settings/api)
   - Gemini (free, for the semantic rerank): [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

2. In this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # pinned deps + pytest/ruff + this package (editable)
copy .env.example .env                # then put your keys in .env
```

## Run the web app

```powershell
python main.py
```

| URL | What |
|---|---|
| http://127.0.0.1:8000 | Web page: ask a question, watch live progress, see results |
| http://127.0.0.1:8000/docs | Interactive API docs (try every endpoint in the browser) |
| `POST /api/search` · `POST /api/agent` | JSON in, JSON out |
| `GET /api/search/stream` · `GET /api/agent/stream` | Live progress as Server-Sent Events (`step`, `result`, `failure`) |
| `GET /health` | Server status |

The server restarts automatically when you edit code (`API_RELOAD`); stop it with Ctrl+C.
It only listens on your own computer (`API_HOST=127.0.0.1`).

### How the web app is built

```
browser ──HTTP──► uvicorn (server) ──► FastAPI app (api/app.py) ──► pipeline / agent
   ▲                                          │
   └──────── SSE: step, step, …, result ◄─────┘  (on_step callback → queue → stream)
```

- **FastAPI** validates requests with Pydantic (bad input → 422) and generates `/docs`.
- **Lifespan:** one shared OpenAlex client (connections + cache) is created at startup.
- **Errors map to HTTP codes:** OpenAlex budget used up → 429, malformed LLM output → 502.
- If the browser disconnects mid-run, the job is cancelled so it stops spending budget.

## Run from the terminal (CLI)

```powershell
research-agent                                   # asks for a question
research-agent "CRISPR delivery since 2020"      # question as an argument
research-agent --agent "find 'attention is all you need' and 3 papers citing it"
research-agent --no-cache "..."                  # fresh results (skip both caches)
research-agent --clear-cache                     # delete cached responses
python -m research_agent ...                     # same as research-agent
```

Run commands from this folder (that's where `.env` is read from).

## Daily commands (venv activated)

| Task | Command |
|------|---------|
| Run tests | `pytest` |
| Lint / auto-fix | `ruff check .` / `ruff check . --fix` |
| Format code | `ruff format .` |
| Add a dependency | add `package==version` to `requirements.txt` (and the loose name to `pyproject.toml` `dependencies`), then `pip install -r requirements-dev.txt` |
| Change a setting | add it to `.env`, e.g. `DEFAULT_RECENT_YEARS=5` (all settings: `src/research_agent/config.py`) |

---

## How the pipeline works

```
question
  -> LLM #1  query plan (keywords + time-window hints)        planning.py
  -> recent window: user range > field pace > default         retrieval/recency.py
  -> OpenAlex: relevant / recent / highly_cited buckets       retrieval/buckets.py
  -> dedupe (Python)                                          retrieval/selection.py
  -> semantic rerank (Gemini embeddings, cosine similarity)   retrieval/rerank.py
  -> balanced selection (Python, no LLM)                      retrieval/selection.py
  -> LLM #2  grounded synthesis (validated JSON)              synthesis.py
```

### Retrieval details (Step 4)

OpenAlex returns only papers containing **every** query word, so query length matters.

| Bucket | Query | Searches in | Sorted by |
|---|---|---|---|
| relevant | `search_query` (3-6 keywords) | title + abstract + full text | relevance |
| recent | `core_query` (2-4 words) | title + abstract, window start → today | citations |
| highly_cited | `core_query` + up to 2 `alternative_queries` | title + abstract | citations (merged) |

- Title + abstract only for the citation-sorted buckets → famous papers that merely *mention* the topic don't win.
- Short core query → one padding word can't knock out a landmark paper.
- Alternative phrasings ("fake news" for "misinformation") → landmarks written in other vocabulary are found.
- Dedupe also merges records with the same title (preprint + conference version).

### Semantic rerank (Gemini embeddings)

Short queries find landmarks, but keywords can't tell "diffusion **models**" (ML) from
"diffusion of innovations" (sociology). So after retrieval, a second stage scores **meaning**:

```
question  --gemini-embedding-001 (RETRIEVAL_QUERY)-->    vector q
paper     --gemini-embedding-001 (RETRIEVAL_DOCUMENT)--> vector d   (title + abstract)
similarity = cosine(q, d)  ->  drop papers below RERANK_MIN_SIMILARITY  ->  selection
```

| Piece | Where |
|---|---|
| `Embedder` interface + `GeminiEmbedder` (batching, per-text cache, retries) | `embeddings.py` |
| score + threshold filter (always keeps at least 12) | `retrieval/rerank.py` |
| relevant bucket / leftover slots ranked by similarity | `retrieval/selection.py` |

- No vector database: ~40 vectors per question are compared in memory. A vector DB is for
  searching **millions** of stored vectors (Phase 4, full-text RAG).
- Embeddings are cached in `embeddings.sqlite` for 90 days, even in evals: the same text
  always gives the same vector, so caching adds no bias.
- Free tier: **100 texts per minute** (each text in a batch counts). A cold question embeds
  ~40 papers, so a cold eval is slow; on 429 the client waits exactly as long as Google asks.
- No `GEMINI_API_KEY` (or `RERANK_ENABLED=false`) → the step is skipped with a notice.

`pipeline.py` glues these together without any `print`/`input`, so the same code can
later power an API, a UI or a LangGraph node. `cli.py` is the only terminal code.

### The "recent" window is dynamic

| Priority | Source | Example |
|---|---|---|
| 1 | **You** state a time range | "… since 2020" → 2020-present; "… last 5 years" → computed in Python |
| 2 | **Field pace** (LLM judgement) | LLM agents → 1-2 years; most ML → 3; clinical medicine → 4-6 |
| 3 | **Default** | `DEFAULT_RECENT_YEARS` (3) |

The LLM only *proposes*; Python *decides*. A user time range is trusted only if the
LLM's quoted phrase really appears in your question, and the field-pace suggestion
is clamped to `MIN_RECENT_YEARS`..`MAX_RECENT_YEARS`.

### Speed and reliability (Step 3)

| Feature | Where | What it does |
|---|---|---|
| **Async** | `httpx.AsyncClient`, `AsyncGroq` | the 3 bucket searches (and parallel agent tool calls) run at the same time |
| **Concurrency limit** | `asyncio.Semaphore` | at most `HTTP_MAX_CONCURRENCY` requests in flight (polite to OpenAlex) |
| **Retries** | `tenacity` | timeouts, 5xx and short 429s are retried with exponential backoff + jitter; an exhausted daily budget fails fast; other 4xx are never retried |
| **OpenAlex cache** | `cache.py` (SQLite) | identical requests are answered from disk: instant and free |
| **LLM cache** | `llm.py` | identical prompts reuse the stored answer. Needed because LLMs aren't deterministic: without it, the query changes every run and the OpenAlex cache never hits |

Caches live in `%LOCALAPPDATA%\research-agent\` (outside OneDrive), expire after
`CACHE_TTL_HOURS` (24), and are skipped with `--no-cache` or `CACHE_ENABLED=false`.

### Package map

| Module | Job | Concept it teaches |
|---|---|---|
| `config.py` | all settings from `.env` | pydantic-settings, typed config |
| `models.py` | `Paper`, `QueryPlan`, `ResearchSummary`, … | Pydantic models instead of dicts |
| `llm.py` | Groq client + `ask_json()` | validating LLM output |
| `prompts.py` | all system prompts | prompts as code |
| `cache.py` | on-disk response cache | caching, cache keys, TTL |
| `sources/openalex.py` | the only OpenAlex code | async HTTP, retries, testable with a fake server |
| `embeddings.py` | Gemini embeddings behind an `Embedder` interface | embeddings, cosine similarity, honouring server retry delays |
| `planning.py` | LLM #1 | structured planning output |
| `retrieval/` | window, buckets, selection | deterministic retrieval logic |
| `synthesis.py` | LLM #2 | grounded summarization |
| `pipeline.py` | runs the steps | orchestration separate from I/O |
| `agent/` | tools + agent loop | tool calling |
| `cli.py` | terminal interface | thin interface layer |

### Config files

- `requirements.txt` — exact runtime versions (`==`) for this environment
- `requirements-dev.txt` — dev tools + `-e .` (installs the package in editable mode)
- `pyproject.toml` — package definition (`[project]`), the `research-agent` command, ruff/pytest settings

---

## Evals — measuring result quality

Unit tests (`pytest`) check the code is **correct**. Evals check the results are **good**.

```powershell
python evals/verify_golden.py                      # check the answer key, write golden.lock.json
python evals/run_eval.py --label baseline          # 13 topics x 3 runs (retrieval mode)
python evals/run_eval.py --topics rag,crispr --runs 1   # quick check
python evals/run_eval.py --mode full --runs 1      # + summary grounding (slow on free tier)
python evals/compare.py evals/results/OLD.json evals/results/NEW.json
```

| File | Role |
|---|---|
| `evals/golden.yaml` | answer key you edit: topics + must-include papers (`foundational` / `recent`) |
| `evals/golden.lock.json` | generated: every OpenAlex ID / DOI / title each paper is known by |
| `evals/results/*.json` | one file per eval run: scores, settings, git commit, every paper + judge verdict |
| `src/research_agent/evals/` | golden matching, LLM judge, metrics, runner (unit-tested) |

| Metric | Meaning |
|---|---|
| `recall` | must-have papers that made the final 12 |
| `candidate_recall` | must-haves retrieved at all |
| `shortlist_recall` | must-haves that survived the semantic rerank (`dropped_in_rerank` / `dropped_in_selection` in the results file say which step lost a paper) |
| `precision` | final papers the LLM judge rates relevant |
| `recent_precision` | same, for papers picked by the "recent" bucket |
| `grounding` | summary citations that really are in the reading list (`--mode full`) |
| `spread` | run-to-run variation; a change smaller than this is noise |

The judge uses a stronger model (`EVAL_JUDGE_MODEL`, default `openai/gpt-oss-120b`) and caches
each verdict per (question, paper). Spot-check `judge_reason` in results files — an LLM judge is
cheap and scalable, not infallible.

---

## Lessons (original scripts in `lessons/`)

### Lesson 1 — basic LLM JSON (`lessons/main.py`)

```
Python → Groq LLM → Prompt → Structured JSON
```

### Lesson 2 — paper search pipeline (`lessons/paper_search.py`)

```
User → LLM (search query) → OpenAlex (papers) → LLM (summary) → Final answer
```

Key idea: the LLM **does not** call OpenAlex by itself. Your **Python code** is the
orchestrator. That pattern is the start of an **agent**: model plans → code uses a tool →
model explains.

### Stage 2 / Lesson 3 — tool calling (`lessons/tool_agent.py`)

```
User → LLM → tool selection → tool call → Python runs it → results → LLM → … → answer
```

| Concept | Meaning |
|---------|---------|
| **Tools** | Python functions: `search_papers`, `get_paper`, `get_citations`, `get_references` |
| **Tool schemas** | JSON descriptions — the only thing the model sees about a tool |
| **Tool selection** | Model chooses which function to call |
| **Tool arguments** | Model fills JSON args like `{"query": "..."}` |
| **Tool results** | Your code runs OpenAlex and returns JSON text |
| **Agent loop** | repeat LLM ↔ tools until a final answer |

Run any lesson: `python lessons\paper_search.py` (they still work).

### About OpenAlex

- Free scholarly metadata API: [https://openalex.org](https://openalex.org)
- Usage is metered against a daily budget. A free API key gives you $1/day of your own
  (a search ≈ $0.001; single-paper lookups are free). Without a key you share a tiny
  budget with everyone on your network → 429 errors.

---

## Mental model

**Lesson 1:** one LLM call for structured extraction.
**Lesson 2:** Python orchestrates OpenAlex, then an LLM summarizes.
**Stage 2:** the LLM chooses tools dynamically inside an agent loop.
**Step 2 (package):** same ideas, organized like production code — typed settings,
validated models, one client per external service, logic separated from I/O, tests.
