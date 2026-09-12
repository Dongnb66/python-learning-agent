# Python Learning Agent · Personalized Learning Multi-Agent System

> A conversational learning-profiling and multi-agent tutoring system built with **LangGraph + FastAPI** — a Python rewrite of the [A3 Software Cup entry](https://github.com/Dongnb66/a3-learning-agent) (Node.js version).

English | **[中文](README.md)**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-7F77DD)
![License](https://img.shields.io/badge/License-MIT-green)

## What it is

A student tells the system — in **natural language** — their major, learning goals, weak points, available study time, and more. The system then runs a **LangGraph-orchestrated pipeline** in a closed loop:

**Load memory → extract a 6-dimension profile → generate a personalized plan → recommend real resources via RAG → create a self-test → produce a study review → (conditional edge) run an autonomous tutoring loop → save memory.**

`load_memory / save_memory` give the system cross-session memory (with a "last time vs. this time" comparison), while `tutor` is a **ReAct-style autonomous agent** — which tools to call, how many rounds, and when to stop are all decided by the model at runtime.

This is a Python migration of the A3 Node.js project: **business logic kept 1:1, tech stack switched to what AI-Agent roles actually ask for** (Python · LangGraph · FastAPI · RAG · SQLAlchemy · Docker).

## Key features

- **Conversational profiling (ProfileAgent)**: extracts a 6-dimension profile — knowledge base / learning goal / cognitive style / weak points / resource preference / study time — and uses regex to **confidently extract name / major** on top of the LLM result
- **Planning (PlannerAgent)**: builds a progressive learning path (with per-step time estimates) from the profile
- **Resource recommendation (ResourceAgent) · RAG anti-hallucination (three code-level guarantees)**: not "telling the model in the prompt not to make things up", but making "no fabrication" a **code invariant** — ① **thresholded retrieval**: BM25 with a relevance floor, so an unmatched topic returns *nothing* instead of padding with zero-score docs; ② **code-level refusal**: when retrieval is empty the node returns `[]` **without calling the model at all** (no evidence → no generation); ③ **URL whitelist**: every item the model returns must match this round's retrieved candidates (or a corpus title), otherwise it is dropped — title-only items are normalized to the real corpus URL. **Even if the model hallucinates, it cannot get through.** Proven by `tests/test_resource_guard.py`
- **Self-test (QuizAgent) + review (ReviewAgent)**: closed-loop learning feedback
- **Autonomous tutoring (TutorAgent) · ReAct tool-calling loop**: once the review surfaces weak points, the model **decides for itself** what to look up — 5 read-only tools (RAG retrieval / read profile / read last session / count sessions / load skill body on demand), looping through *decide → act → observe → decide again* until it has enough, capped at 4 rounds. Every decision and observation is recorded as an auditable `trace`. Falls back to a rule policy when no key is configured or the model errors out
- **Skill lazy loading (progressive disclosure)**: teaching methods are packaged as skill files `app/data/skills/*.md` (frontmatter metadata + body playbook). Only **metadata stays resident** in the system prompt (`list_skills` scans the frontmatter only, never the body); the **body is loaded on demand** — when the model judges a scenario matches a skill, it autonomously calls the `load_skill` tool inside the ReAct loop. Resident context grows linearly with the *number* of skills, not their length. Unknown skill names are **rejected at the code level** (returns the available list instead of fabricating), the same anti-hallucination principle as retrieval. Verified by `tests/test_skills.py`
- **LangGraph orchestration**: state graph `load_memory → profile → planner → resource → quiz → review → tutor → save_memory`, with a **conditional edge** after `review` — the tutoring loop only runs when weak points exist. (Deterministic where it should be, autonomous where it must be)
- **Swappable provider**: DeepSeek by default (OpenAI-compatible); switch to OpenAI / Claude / Qwen / Bailian MaaS by editing 3 lines
- **Type-safe**: Pydantic + type hints + auto-generated OpenAPI docs
- **Testable + evaluable**: 146 tests run with no real LLM calls and no API key; plus an **anti-hallucination evaluation suite** (`eval/bad_cases.json` + `scripts/run_eval.py`) — 6 normal + 3 adversarial cases / 5 assertion types quantifying "0 fabricated links, 100% verifiable sources", ready for CI
- **Observable**: `/api/metrics` (JSON or Prometheus text) + `/api/traces/{id}` — per-node timings for all 8 nodes, model/tool call counts, failure reasons; **the guard actions themselves are metrics**, so "no hallucinations" is a number you can check rather than a claim
- **Reproducible**: `python scripts/reproduce.py` runs tests + normal eval + adversarial eval + ablation in one command, with an **environment fingerprint** and a non-zero exit gate
- **One-command Docker**: `docker-compose up`

## Architecture

```mermaid
flowchart TD
    User[Student natural-language chat] --> API[FastAPI entry]
    API --> SG[LangGraph state graph]
    subgraph WF[Multi-agent workflow]
        direction LR
        A[ProfileAgent<br/>6-dim profiling] --> B[PlannerAgent<br/>Learning path]
        B --> C[ResourceAgent<br/>RAG resources]
        C --> D[QuizAgent<br/>Self-test]
        D --> E[ReviewAgent<br/>Study review]
    end
    SG --> WF
    E -->|conditional edge: weak points| T[TutorAgent<br/>autonomous ReAct loop]
    E -->|no weak points| SM[save_memory]
    T -.autonomously calls.-> TOOLS[5 read-only tools]
    T --> SM
    SM --> DB2[(SQLite<br/>profiles + learning_sessions)]
    C <-.retrieve real docs.-> KB[(BM25 local corpus)]
    C -.no match.-> RC[Code-level refusal<br/>empty resources, no LLM call]
    C -.whitelist.-> WL[Drop non-corpus links]
    A --> DB[(SQLite profile store)]
    WF --> LLM[DeepSeek / OpenAI-compatible endpoint]
    T --> LLM
```

> `WF` is a **deterministic workflow** (execution path fixed in code, so teaching paths stay reproducible);
> `TutorAgent` is an **autonomous agent** (which tools, how many rounds, when to stop are decided by the model at runtime).
> The two deliberately coexist — deterministic where it should be, autonomous where it must be.
>
> `ResourceAgent`'s three anti-hallucination guarantees (refusal / whitelist) live in **code**, not in the model's good intentions.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI + Pydantic | Type safety + auto OpenAPI docs |
| Agent orchestration | LangGraph | Industrial-grade multi-agent framework with visual state |
| LLM access | LangChain `ChatOpenAI` + `function calling` structured output | Works with DeepSeek / OpenAI / Claude; reliably parses to Pydantic |
| RAG retrieval | BM25 (langchain-community + rank-bm25) | Offline, no embedding API needed; enough to demo anti-hallucination |
| Storage | SQLAlchemy 2.0 + SQLite | Zero-dependency; swap to PostgreSQL without touching business code |
| Deployment | Docker + docker-compose | One-command startup |

## Quick start

```bash
# 1. Clone
git clone https://github.com/Dongnb66/python-learning-agent.git
cd python-learning-agent

# 2. Configure API key
cp .env.example .env
# Edit .env and set LLM_API_KEY (DeepSeek by default)

# 3a. Run locally
pip install -e .
uvicorn app.main:app --reload --port 8000

# 3b. Or with Docker
docker-compose up
```

Open http://localhost:8000/docs for the interactive API docs.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/profile/build` | Build the student profile only |
| POST | `/api/learn` | Run the full pipeline (profile → plan → resources → quiz → review) |
| GET | `/api/profile/{user_id}` | Load a saved profile from the database |

### Try it

```bash
curl -X POST http://localhost:8000/api/learn \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "student-001",
    "messages": [
      {"role": "user", "content": "I am a software-engineering student, average Python, weak at algorithms"},
      {"role": "user", "content": "I want to learn AI Agents, ~10 hours/week, prefer video"}
    ]
  }'
```

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

- `tests/test_profile.py`: profiling + name/major regex (mocked LLM)
- `tests/test_pipeline.py`: **end-to-end multi-agent pipeline** (mocks all LLMs, asserts every artifact + conditional-edge tutoring)
- `tests/test_memory.py`: cross-session memory (read-back / accumulate / failure isolation)
- `tests/test_tutor_agent.py`: autonomous tutoring agent (ReAct loop, self-termination, round cap, privilege guard, graceful degradation)
- `tests/test_resource_guard.py`: **the three anti-hallucination guarantees** (empty retrieval / no LLM call on refusal / whitelist drops fabricated links)
- `tests/test_skills.py`: skill lazy loading (metadata catalog never leaks body text / body readable on demand / unknown names refused)
- `tests/test_eval_suite.py`: evaluator self-check (judgement logic + full offline case run + `expect_refusal` assertiveness + adversarial set must refuse)
- `tests/test_telemetry.py`: observability layer (span timing / error flagging / nesting, trace collection and bounded storage, Prometheus rendering, model-call counting proxy, and the guardrail counters)
- `tests/test_metrics_api.py`: metrics & trace endpoint contracts (`X-Trace-Id` round-trip, all 8 node spans, model/tool call counts, `guardrails` block, Prometheus format)
- `tests/test_reproduce.py`: the reproduce script itself (fingerprint fields complete; the gate really rejects failing stages)

**146 passed.** No real LLM is called — no API key needed.

## Evaluation (anti-hallucination suite)

"I built RAG so it won't hallucinate" is an unverifiable claim. So it is broken down into **decidable assertions**:

```bash
python scripts/run_eval.py          # zero-config: falls back to the offline stub, runs in-process, no server needed
python scripts/run_eval.py --cases eval/adversarial_cases.json   # adversarial set: must refuse
```

It runs the 6 cases in `eval/bad_cases.json` against the whole pipeline with 5 assertion types:
artifact completeness; **anti-hallucination rate** (every recommended URL must exist in the local corpus — fabricated links must be 0); correct refusal on unknown topics; **hard refusal** (`expect_refusal` — must be empty, unlike `may_refuse_if_no_match` which merely allows it); and deterministic profile fields.

It prints a report and writes `eval/report_<timestamp>.json` (including the anti-hallucination rate); a non-zero exit code means failure, so it drops straight into CI.

Current result: **6/6 passed, 24 resources recommended, 0 fabricated links, 100% anti-hallucination rate**;
**adversarial 3/3 refused, 0 resources, 100% refusal rate**.

### Two holes that the assertions themselves were hiding (fixed)

**Hole 1 — `expect_refusal` was never implemented.** The adversarial set declared "this case must refuse" from day one, but the evaluator only implemented "may refuse" — empty or not, it passed. So even with all three guards disabled the adversarial cases still "passed": the mock returns resources whose URLs all come from the real corpus, so `resource_url_must_be_local` finds nothing wrong. **An assertion that never fires is not an assertion.**

**Hole 2 — even implemented, the assertion could not reach its target.** The retrieval query is the **step titles of the learning plan**, not the user's words. In offline-stub mode the plan is a fixed stub (always about Python/FastAPI), so typing "quantum computing" still hits the corpus — the refusal branch is never reached.

Fix: ① implement `expect_refusal` in the evaluator; ② let adversarial cases declare `plan_titles`, and have `eval_suite.GraphInvoker` inject a planner node that emits exactly that plan, so the adversarial topic actually reaches the retrieval layer (the injected plan lives in the case data — visible and auditable, not a back door hidden in the validation logic).

Verified effective: disabling the *refusal* and *URL whitelist* guards **together** flips the adversarial set from 3/3 to 0/3 with `expected refusal ... but returned 4 resources`. Disabling only one still passes — the correct behaviour of defence in depth. Both assertions are locked in `tests/test_eval_suite.py` and run in CI.

> Note: with the offline stub the *refusal* branch is not exercised by the normal cases (fixed plan, retrieval always hits); that branch is covered by the deterministic unit tests in `tests/test_resource_guard.py` — **empty retrieval → empty resources and zero model calls**.

## Observability (what is actually happening in production)

Offline evaluation answers "is it correct before commit"; observability answers "how is it behaving once running" — which nodes a request visited, which one is slow, how many model calls it made, and how often the anti-hallucination guards fired.

```bash
curl http://localhost:8000/api/metrics                          # JSON
curl http://localhost:8000/api/metrics?format=prometheus
curl http://localhost:8000/api/traces/<trace_id>
```

Instrumentation lives in the **orchestration layer** (nodes are wrapped once in `app/graph.py`), so agent code stays clean; a counting proxy at the `app/llm.py` factory exits covers all six model call sites automatically, including the ReAct branch after `bind_tools`. Every `POST /api/learn` returns `X-Trace-Id`.

| Design point | Why |
|---|---|
| Zero-dependency, stdlib only | Must run offline with no API key; naming follows OTel / Prometheus so swapping in an exporter later changes nothing else |
| `contextvar`-carried trace | One trace per request, no cross-talk under concurrency, and the trace never pollutes `AgentState` |
| Bounded trace store (last 50) | Keeping all history in memory is a slow leak; long-term retention belongs to a log/timeseries system |
| Mock vs real calls counted separately | `llm_calls_total` counts all, `llm_mock_calls_total` the mocked subset — so the counter is testable offline yet never mistaken for real API usage |
| **Guard actions as metrics** | The `guardrails` block exposes `retrieval_empty_total` / `refusals_total` / `fabricated_links_blocked_total`, turning "I do anti-hallucination" into numbers you can check |

## One-command reproduction

```bash
python scripts/reproduce.py                # tests + normal eval + adversarial eval + ablation
python scripts/reproduce.py --skip-pytest  # what CI runs (tests already ran in the previous step)
```

Writes `eval/reproduce_report.json` (**committed**, so every number in this README can be checked against it) with an environment fingerprint: Python version, platform, dependency versions, git commit, working-tree cleanliness and mock mode. A non-zero exit code on any failing stage makes it usable as a quality gate.

## Relation to A3 (Node.js original)

Both repos share the same business design; this repo is the Python rewrite:

| Dimension | A3 Node.js | Python (this repo) |
|---|---|---|
| Agent design | ✓ 5 agents | ✓ ported + autonomous tutor agent |
| 6-dim profiling | ✓ | ✓ aligned + name/major regex |
| RAG anti-hallucination | ✓ | ✓ code-level: thresholded retrieval / refusal on empty hits / URL whitelist |
| Cross-session memory | × | ✓ 3-layer (AgentState / profiles / learning_sessions) |
| Autonomous decisions | × | ✓ ReAct tool-calling loop |
| Effect evaluation | × | ✓ anti-hallucination suite (6 normal + 3 adversarial cases / 5 assertion types / JSON report / CI-ready) |
| Observability | × | ✓ `/api/metrics` (node timings + guardrail counters) · `/api/traces/{id}` round-trip |
| Reproducibility | × | ✓ `scripts/reproduce.py` one command + environment fingerprint + non-zero exit gate |
| Engineering | Express + React | FastAPI + optional frontend |
| Agent framework | hand-rolled orchestrator | **LangGraph** (with conditional edges) |
| Docker deployment | × | ✓ |

## Roadmap

- [x] 5 agents + LangGraph state graph
- [x] Cross-session 3-layer memory
- [x] Autonomous tutoring agent (ReAct loop with tools)
- [x] RAG anti-hallucination resource recommendation
- [x] FastAPI endpoints + SQLite persistence
- [x] Anti-hallucination evaluation suite + CLI (runs offline, incl. adversarial set and hard `expect_refusal`)
- [x] Observability: node-level tracing + `/api/metrics` + `/api/traces/{id}` + guardrail counters
- [x] Reproducibility: one-command reproduce script (with environment fingerprint) wired into CI
- [x] pytest 146 passed (mocked LLM, no API key needed) + Docker
- [x] Frontend (reuses the A3 React app, 9 pages)
- [ ] Hosted online demo
- [ ] Demo video
- [ ] Vector retrieval + rerank (currently BM25 only)

## License

MIT
