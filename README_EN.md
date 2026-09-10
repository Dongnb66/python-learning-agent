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
- **Resource recommendation (ResourceAgent) · RAG anti-hallucination**: retrieves real materials with BM25 first; the LLM **may only recommend from retrieved, real links**, suppressing fabricated resources at the mechanism level
- **Self-test (QuizAgent) + review (ReviewAgent)**: closed-loop learning feedback
- **Autonomous tutoring (TutorAgent) · ReAct tool-calling loop**: once the review surfaces weak points, the model **decides for itself** what to look up — 4 read-only tools (RAG retrieval / read profile / read last session / count sessions), looping through *decide → act → observe → decide again* until it has enough, capped at 4 rounds. Every decision and observation is recorded as an auditable `trace`. Falls back to a rule policy when no key is configured or the model errors out
- **LangGraph orchestration**: state graph `load_memory → profile → planner → resource → quiz → review → tutor → save_memory`, with a **conditional edge** after `review` — the tutoring loop only runs when weak points exist. (Deterministic where it should be, autonomous where it must be)
- **Swappable provider**: DeepSeek by default (OpenAI-compatible); switch to OpenAI / Claude / Qwen / Bailian MaaS by editing 3 lines
- **Type-safe**: Pydantic + type hints + auto-generated OpenAPI docs
- **Testable**: all LLM calls are mockable — the test suite runs with no API key
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
    C <-.retrieve real docs.-> KB[(BM25 local corpus)]
    A --> DB[(SQLite profile store)]
    WF --> LLM[DeepSeek / OpenAI-compatible endpoint]
```

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
- `tests/test_pipeline.py`: **end-to-end multi-agent pipeline** (mocks all 5 LLMs, asserts every artifact is produced)

No real LLM is called — no API key needed.

## Relation to A3 (Node.js original)

Both repos share the same business design; this repo is the Python rewrite:

| Dimension | A3 Node.js | Python (this repo) |
|---|---|---|
| Agent design | ✓ 5 agents | ✓ ported + autonomous tutor agent |
| 6-dim profiling | ✓ | ✓ aligned + name/major regex |
| RAG anti-hallucination | ✓ | ✓ BM25-constrained recommendation |
| Cross-session memory | × | ✓ 3-layer (AgentState / profiles / learning_sessions) |
| Autonomous decisions | × | ✓ ReAct tool-calling loop |
| Engineering | Express + React | FastAPI + optional frontend |
| Agent framework | hand-rolled orchestrator | **LangGraph** (with conditional edges) |
| Docker deployment | × | ✓ |

## Roadmap

- [x] 5 agents + LangGraph state graph
- [x] Cross-session 3-layer memory
- [x] Autonomous tutoring agent (ReAct loop with tools)
- [x] RAG anti-hallucination resource recommendation
- [x] FastAPI endpoints + SQLite persistence
- [x] pytest 30 passed (mocked LLM, no API key needed) + Docker
- [ ] Frontend (reuse the A3 React app)
- [ ] Hosted online demo
- [ ] Demo video

## License

MIT
