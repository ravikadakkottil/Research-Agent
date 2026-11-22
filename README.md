# Research Agent

Ask a question about any research topic and get a **balanced reading list**
(foundational, influential and recent papers) plus a **grounded summary**
that only cites papers from that list.

```
"diffusion models for image generation"
   → 12 papers: Denoising Diffusion Probabilistic Models (2020), Latent Diffusion (2022), …,
     the most-cited recent work
   → summary: foundational findings · recent developments · how the field evolved · gaps
```

Built with Python, LangGraph, FastAPI, Groq-hosted LLMs, the [OpenAlex](https://openalex.org)
scholarly database and Gemini embeddings. Retrieval quality is measured with an evaluation
suite (golden dataset + LLM judge).

---

## Features

- **Multi-perspective retrieval:** three parallel searches (relevance, recent, highly
  cited), so a reading list covers both landmark papers and current work.
- **Adaptive "recent" window:** respects time ranges in the question ("since 2020",
  "last 5 years"). Otherwise it adapts to how fast the field moves (LLM research: 1–2 years;
  clinical medicine: 4–6).
- **Citation expansion:** landmark papers that don't use the topic's keywords are found
  through the reference lists of the best results. *The spread of true and false news
  online* never says "misinformation", but the field's top papers all cite it.
- **Second source for biomedical topics:** Europe PMC (PubMed abstracts + full text) runs
  alongside OpenAlex and fills in abstracts OpenAlex lacks, so landmarks whose titles don't
  name the topic are still found. It is optional and fail-soft: a slow or failed response
  never blocks a question.
- **Semantic rerank:** candidates are scored by meaning against the question with
  `gemini-embedding-001`, so keyword collisions ("diffusion **models**" vs "diffusion of
  innovations") are filtered out.
- **Grounded synthesis:** the summary is validated JSON, and it may cite only papers that
  were actually retrieved.
- **Tool-calling agent mode:** the LLM chooses its own tools (search, paper lookup,
  citations, references) for open-ended questions. Every paper it cites is checked against
  what the tools actually returned. The answer comes with a verified Sources list, and
  unknown IDs are flagged. Answers render as Markdown with math (KaTeX), sanitized
  before display.
- **Tracing (optional):** with [Langfuse](https://langfuse.com) keys set, every question
  becomes a trace: each step's inputs and outputs, LLM prompts, replies and token counts,
  embedding calls, and every OpenAlex query (with cache hits). Eval runs attach their scores
  (recall, precision, …) to their traces and share a session, so weak runs can be filtered and
  opened step by step. API keys never appear in traces.
- **Web app + API:** live progress streamed over Server-Sent Events, plus interactive
  OpenAPI docs.
- **Resilient and efficient:** async I/O, bounded concurrency, retries with backoff that
  honour server rate-limit hints, and on-disk caches for API and LLM responses.
- **Evaluated:** recall and precision on a 13-topic golden set across AI, biology, medicine,
  materials science and social science.

## Results

Retrieval quality on the golden set (13 topics × 3 runs, judged by `gpt-oss-120b`):

| Version | Recall | Precision | Recent-paper precision |
|---|---|---|---|
| Baseline (single keyword query) | 0.42 | 0.49 | 0.19 |
| + title/abstract scope, citation-sorted recent bucket | 0.19 | 0.72 | 0.60 |
| + short core query and alternative phrasings | 0.53 | 0.65 | 0.41 |
| + semantic rerank | 0.58 | 0.79 | 0.62 |
| + citation expansion | 0.71 | 0.76 | 0.58 |
| + similarity-ordered citation picks | 0.78 | 0.80 | 0.58 |
| + Europe PMC for biomedical topics | 0.74 | 0.81 | 0.60 |
| LangGraph orchestration, same retrieval | 0.79 | 0.78 | 0.55 |
| **+ recent picks ranked by similarity (current)** | **0.74** | **0.83** | **0.72** |

Citation expansion and Europe PMC raised the share of must-include papers that are retrieved
at all from 0.65 to 0.93–1.00. Repeated evals of unchanged code differ by up to ±0.04
(planning is an uncached LLM call), so the Europe PMC and LangGraph rows reflect the same
retrieval quality.

Recent picks used to be the newest papers in the window. With a generic core query, those
were popular but general papers. Ranking them by similarity to the question raised
recent-paper precision from 0.55 to 0.72 and overall precision from 0.78 to 0.83. The latest
row is based on 28 of 39 runs, because the LLM provider's daily token limit cut off the
rest. Its recall change (−0.04 on the same runs) is within noise.

## How it works

```
question
  → LLM: search plan (relevance query, short core query, alternative phrasings, time window)
  → recent window: user's range > field pace > default
  → OpenAlex, in parallel:
        relevant      full-text search, ranked by relevance
        recent        title+abstract, inside the window, ranked by citations
        highly cited  title+abstract, core query + alternatives, ranked by citations
        + Europe PMC  biomedical topics only, ranked by citations, mapped onto OpenAlex by DOI
  → dedupe (DOI / OpenAlex ID / title)
  → semantic scoring (Gemini embeddings, cosine similarity to the question)
  → citation expansion: papers cited by ≥ 3 of the top 15 candidates
  → similarity threshold (drops off-topic papers; stricter for citation-only papers)
  → balanced selection (quotas per perspective; recent and co-cited picks ranked by similarity)
  → LLM: grounded synthesis (validated JSON)
```

The steps run as a [LangGraph](https://langchain-ai.github.io/langgraph/) graph, one node
per step. Optional steps (semantic scoring, citation expansion, similarity filter) are
conditional edges that follow the configuration. `research-agent --graph` prints the graph
as a Mermaid diagram.

## Quick start

Requires Python 3.11+ and three free API keys:
[Groq](https://console.groq.com), [OpenAlex](https://openalex.org/settings/api) and
[Gemini](https://aistudio.google.com/apikey). Gemini is optional: without it the rerank
step is skipped. [Langfuse](https://cloud.langfuse.com) keys are optional too: without them
tracing is off.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Create a file named `.env` in the project folder with your keys:

```ini
GROQ_API_KEY=...            # required
OPENALEX_API_KEY=...        # recommended: without it, requests share a small anonymous budget
GEMINI_API_KEY=...          # optional: semantic rerank (skipped without it)

# Optional: Langfuse tracing (tracing is off without these)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Any setting from [Configuration](#configuration) can be added to `.env` the same way.

### Web app

```bash
python main.py
```

| URL | |
|---|---|
| http://127.0.0.1:8000 | Web UI with live progress |
| http://127.0.0.1:8000/docs | Interactive API docs |

### API

| Endpoint | Description |
|---|---|
| `POST /api/search` | Reading list + summary (JSON) |
| `POST /api/agent` | Tool-calling agent answer (Markdown) + verified `sources` + `unverified_ids` |
| `GET /api/search/stream?question=…` | Same as `/api/search`, streamed as SSE events (`step`, `result`, `failure`) |
| `GET /api/agent/stream?question=…` | Agent, streamed |
| `GET /health` | Health check |

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"question": "CRISPR delivery methods since 2020"}'
```

### CLI

```bash
research-agent "perovskite solar cell stability"
research-agent --agent "find 'Attention Is All You Need' and 3 papers citing it"
research-agent --no-cache "..."      # bypass caches
research-agent --clear-cache
research-agent --graph               # print the pipeline graph (Mermaid)
```

## Configuration

All settings live in `src/research_agent/config.py` and can be overridden in `.env`:

| Setting | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `openai/gpt-oss-20b` | planning + synthesis model (Groq) |
| `DEFAULT_RECENT_YEARS` | `3` | "recent" window when nothing else applies |
| `TARGET_FINAL_PAPERS` | `12` | reading-list size |
| `RERANK_ENABLED` | `true` | semantic rerank on/off |
| `RERANK_MIN_SIMILARITY` | `0.66` | similarity threshold (calibrated on the golden set) |
| `CITATION_EXPANSION_ENABLED` | `true` | citation expansion on/off |
| `EXPANSION_MIN_CITERS` | `3` | top papers that must cite a paper before it's added |
| `EXPANSION_MIN_SIMILARITY` | `0.68` | stricter similarity bar for papers found only through citations |
| `EUROPEPMC_ENABLED` | `true` | Europe PMC search for biomedical questions |
| `EUROPEPMC_TIMEOUT_SECONDS` | `20` | time limit for that step (then OpenAlex only) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | none | Langfuse project keys; tracing is on when both are set |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse region (US: `https://us.cloud.langfuse.com`) or a self-hosted URL |
| `TRACING_ENABLED` | `true` | turn tracing off without removing the keys |
| `CACHE_TTL_HOURS` | `24` | API / LLM response cache lifetime |
| `HTTP_MAX_CONCURRENCY` | `5` | parallel OpenAlex requests |

## Evaluation

```bash
python evals/run_eval.py --label my-change          # 13 topics × 3 runs
python evals/run_eval.py --topics rag,crispr --runs 1
python evals/compare.py evals/results/A.json evals/results/B.json
```

| Metric | Meaning |
|---|---|
| `recall` | must-include papers (from `evals/golden.yaml`) present in the final list |
| `candidate_recall` / `shortlist_recall` | the same, after retrieval / after rerank: locates where papers are lost |
| `precision` | share of the final list an LLM judge rates relevant to the question |
| `recent_precision` / `expansion_precision` | precision of the papers chosen as "recent" / via citation expansion |
| `grounding` | summary citations that exist in the reading list (`--mode full`) |

Each run is repeated to measure natural LLM variance. `compare.py` marks differences
within that spread as noise. With tracing on, each run's metrics are also attached to its
Langfuse trace as scores, and all runs of one eval share a session.

## Development

```bash
pytest                  # unit tests (no network: fake HTTP server, fake LLM/embedder)
ruff check . && ruff format .
```

```
src/research_agent/
  config.py  models.py  prompts.py      settings, data models, prompts
  sources/                              async OpenAlex + Europe PMC clients (retries, cache)
  llm.py  embeddings.py                 Groq JSON calls, Gemini embeddings
  planning.py  synthesis.py             LLM steps
  retrieval/                            recent window, buckets, second source, rerank,
                                        citation expansion, selection
  pipeline.py                           orchestration (LangGraph graph: state, nodes, edges)
  tracing.py                            Langfuse tracing helpers (no-op without keys)
  agent/                                tool-calling agent
  api/                                  FastAPI app + web UI
  evals/                                golden matching, LLM judge, metrics, runner
evals/                                  golden dataset, eval + compare scripts, results
tests/
```

## Roadmap

- [x] Citation-graph expansion (pull in landmark papers that top results cite)
- [x] Second source for biomedical topics (Europe PMC)
- [x] Langfuse tracing (steps, LLM calls, tokens, eval scores)
- [x] LangGraph orchestration of the search pipeline
- [ ] LangGraph agent (tool-calling loop as a graph)
- [ ] Deep-research mode (planner + parallel sub-researchers)
- [ ] Full-text RAG over open-access PDFs

## Acknowledgements

Paper metadata from [OpenAlex](https://openalex.org) (CC0) and
[Europe PMC](https://europepmc.org).
