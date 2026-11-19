# Project Handoff — GenAI / Agentic Research Learning (L1)

Paste this into a new chat/agent so context is not lost.

---

## UPDATE 2026-11-18 — direction changed (this section overrides older notes below)

- **New goal:** build a full research agent using **production frameworks and tools used in industry**, while still learning (keep explaining what/why). The "no LangChain/LangGraph / no vector DB" constraints below are **lifted**, to be introduced one phase at a time.
- **Tooling:** pip + venv (uv not usable: can't modify PATH). `requirements.txt` pinned, `requirements-dev.txt` adds pytest/ruff + `-e .`. Git initialized, **nothing committed yet** (user commits manually).
- **Done:** Step 1 (git, pyproject, ruff, pytest); Step 2 (package `src/research_agent/`, Pydantic models + settings, single OpenAlex client with `OPENALEX_API_KEY`, CLI `research-agent`, dynamic recent window: user range > LLM field pace > default). Original scripts moved to `lessons/` unchanged (except the OpenAlex API key fix). Web app: `python main.py` → FastAPI (`src/research_agent/api/`), SSE streaming, web page at /, docs at /docs. Step 3 (async httpx/AsyncGroq, tenacity retries, semaphore, SQLite caches for OpenAlex AND LLM responses in %LOCALAPPDATA%\research-agent, `--no-cache`/`--clear-cache`).
- **Plan next:** Step 4 retrieval quality (recent bucket sort by citations/FWCI, multi-query, semantic rerank, Semantic Scholar/arXiv) → Step 5 evals → Phase 2 LangGraph + Langfuse → Phase 3 deep-research mode (supervisor + subagents) → Phase 4 full-text RAG (per-query vector store, cached by paper ID) → Phase 5 FastAPI/UI.
- **OpenAlex:** now metered; free key = $1/day (search ≈ $0.001, single lookups free).
- **Known quality issues (Step 4):** recent bucket sorted by `publication_date:desc` returns off-topic/future-dated papers; highly_cited bucket drifts off-topic; LLM query can be overly long.

---

## Who / intent

- **Learner:** beginner to GenAI and Agentic AI.
- **Goal:** build understanding step-by-step (not ship a production app).
- **Teaching style required:** explain *what* and *why* for each piece (LLM, prompt, JSON, tools, agent loop). Keep code simple and educational.
- **Stack preference:** Python + Groq API + OpenAlex. Prefer plain Python over frameworks.

---

## Use case being built

An **agentic academic research assistant**:

1. User asks a research question in the terminal.
2. System finds relevant scholarly papers (OpenAlex).
3. System synthesizes a grounded answer (not inventing papers).

Long-term direction (learning stages), not all done yet:

```
Stage 0/1 — LLM basics + structured JSON
Stage 1.5 — Python-orchestrated multi-bucket retrieval + summary
Stage 2   — LLM tool calling + agent loop          ← current
Stage 3+  — (not started) possibly better planning, memory, UI, LangChain/LangGraph later
```

---

## Repo location

```
c:\Users\Ravi.k\OneDrive - Neostats Analytics Solutions L.L.C\Documents\Learning\L1
```

### Key files

| File | Role |
|------|------|
| `main.py` | Lesson 1 — single LLM call → structured JSON (product review extraction demo) |
| `paper_search.py` | Lesson 2 — fixed pipeline: LLM query → multi-bucket OpenAlex → Python select → LLM synthesize |
| `tool_agent.py` | Stage 2 — LLM chooses tools dynamically (`search_papers`, `get_paper`, `get_citations`, `get_references`) |
| `README.md` | Human-readable lesson notes |
| `.env` | Secrets (`GROQ_API_KEY`, optional `OPENALEX_MAILTO`) — never commit |
| `.env.example` | Template for env vars |
| `requirements.txt` | `groq`, `python-dotenv`, `requests`, `certifi` |

---

## Config / environment

- **LLM host:** Groq
- **Model currently used in `paper_search.py` and `tool_agent.py`:** `openai/gpt-oss-20b`
  - Originally requested: `llama-3.1-8b-instant` (still fine to switch; keep one MODEL constant)
- **Paper DB:** OpenAlex (`https://api.openalex.org/works`) — free, no API key
- **Env:**
  - `GROQ_API_KEY` (required)
  - `OPENALEX_MAILTO` (optional polite pool)
- **Windows SSL note:** use `verify=certifi.where()` on OpenAlex requests (corporate cert issues)

Run pattern:

```powershell
cd "...\Documents\Learning\L1"
.\.venv\Scripts\Activate.ps1   # if venv exists
pip install -r requirements.txt
python tool_agent.py           # or paper_search.py / main.py
```

---

## What has been built (architecture)

### Lesson 1 — `main.py`

```
Python → Groq LLM → Prompt → Structured JSON
```

Teaches: API key, chat messages (system/user), JSON mode, `json.loads`.

### Lesson 2 — `paper_search.py` (Python orchestrates tools)

```
User
 → LLM #1 generates OpenAlex search query (JSON)
 → OpenAlex multi-bucket retrieval
 → Python dedupe + balanced selection
 → LLM #2 research synthesis (JSON)
 → Final answer
```

**Retrieval concept to preserve:**

> Do not rely only on newest papers or only relevance. Combine relevance + recency + foundational/highly-cited importance.

Buckets:

- **relevant** — `sort=relevance_score:desc` (~10)
- **recent** — OpenAlex date filter last `RECENT_YEARS` (default 3 → e.g. 2024+ in 2026) (~10)
- **highly_cited** — `sort=cited_by_count:desc` with same search (~10)

Then:

- Deduplicate by DOI → OpenAlex ID → normalized title
- Attach `retrieval_reasons: ["relevant"|"recent"|"highly_cited", ...]`
- Deterministic Python selection (~12 papers; soft quotas 4/3/3 + multi-reason fill)
- **No second LLM for ranking** (explicit learning constraint)
- Summary JSON includes foundational vs recent vs evolution fields
- Interactive terminal input; verbose step prints

### Stage 2 — `tool_agent.py` (LLM chooses tools)

```
User → LLM → tool_calls → Python executes OpenAlex tools → results → LLM → … → final answer
```

Tools:

- `search_papers(query, limit?)`
- `get_paper(paper_id)`
- `get_citations(paper_id, limit?)` — papers that cite this work (`filter=cites:`)
- `get_references(paper_id, limit?)` — bibliography via `referenced_works` then hydrate

Learning concepts encoded in comments/prints:

- Tools, tool schemas, tool selection, tool arguments, tool results, agent loop
- `MAX_AGENT_STEPS = 8`
- `tool_choice="auto"`
- Local tool calling (not LangChain, not Groq built-in compound tools)

**Difference vs Lesson 2:** Lesson 2 hardcodes OpenAlex steps in Python; Stage 2 lets the LLM decide which tools/args to use.

---

## Hard constraints (do not violate unless user asks)

When generating new code or the next stage:

1. **Keep it educational and simple** — comments explain WHY.
2. **Do NOT introduce LangChain / LangGraph yet** (unless user explicitly asks).
3. **Do NOT introduce a vector DB yet.**
4. **Do NOT add another LLM ranking step to `paper_search.py`** unless asked.
5. **Do NOT replace OpenAlex** with another paper DB.
6. **Do NOT redesign the whole app** — incremental lessons, often new files over rewriting old ones.
7. Prefer **interactive terminal input** for questions (no hardcoded research question as the only path).
8. Keep **`.env` / Groq / OpenAlex** setup; reuse `certifi` for SSL.
9. Make knobs **configurable constants** near top of file.
10. Ground answers in retrieved papers — **no invented citations**.
11. Type hints + small helper functions preferred over one huge function.
12. Explain changes briefly after edits (what/why), matching learner style.

---

## How to continue (suggested next stages)

Only if user asks; pick one lesson at a time:

1. **Improve Stage 2 agent:** force structured final JSON; add a `search_recent` / `search_highly_cited` tool so tool-calling agent also gets balanced retrieval.
2. **Merge lessons:** tool-calling agent that internally uses multi-bucket retrieval as one tool.
3. **Memory / multi-turn chat** in the terminal (conversation history).
4. **Evaluation:** simple checks that answer cites only returned paper titles.
5. **Later:** LangGraph / planning / UI — only after user is ready.

---

## Prompt template for a new agent chat

Copy-paste:

```text
Continue my GenAI learning project in:
c:\Users\Ravi.k\OneDrive - Neostats Analytics Solutions L.L.C\Documents\Learning\L1

Read HANDOFF.md first.

I am a beginner learning GenAI / Agentic AI.
Stack: Python + Groq (`openai/gpt-oss-20b` currently) + OpenAlex.
Lessons so far:
- main.py = structured JSON
- paper_search.py = multi-bucket retrieval + Python selection + LLM summary
- tool_agent.py = Stage 2 tool calling agent loop

Constraints: keep educational; no LangChain/LangGraph/vector DB unless I ask;
incremental changes; explain concepts; interactive CLI; no invented papers.

Next task I want:
<describe the next feature here>
```

---

## Mental model one-liners (for the learner)

- **LLM** predicts text; it does not magically call APIs.
- **You** (Python) either hardcode tool use (Lesson 2) or run tools when the model emits `tool_calls` (Stage 2).
- **Prompt + schema** steer structure; **tools** give live external data.
- **Agent** ≈ loop: model → optional tools → model → … → stop.
- **Good research retrieval** balances relevance, recency, and foundational importance.
