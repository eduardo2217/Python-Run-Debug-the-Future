# BBC News Summarizer — FastAPI + OpenRouter

A production-style Python service that **summarizes news articles** from the
public **BBC News** dataset through an **OpenAI-compatible language model**,
exposed as a FastAPI endpoint. It ships with robust error handling, retries
with exponential backoff, an in-memory TTL cache, sliding-window rate limiting,
and a Docker build — both contest bonuses are implemented.

---

## 1. Overview

- **Task:** Text summarization (abstractive, 2–3 punchy sentences). The prompt is
  **fixed server-side**: the model detects whether the text is journalistic or
  technical and produces a summary that is entertaining and impactful so the
  reader stays engaged.
- **Framework:** [FastAPI](https://fastapi.tiangolo.com) (OpenAPI docs enabled at `/docs`).
- **AI service:** The official **OpenAI Python SDK** talking to a
  **free OpenRouter** endpoint (an API key is *optional* for free models).
  - Endpoint: `https://openrouter.ai/api/v1` (OpenAI-compatible protocol).
  - Default model: **`meta-llama/llama-3.3-70b-instruct:free`**.
  - Model is fully configurable via `MODEL`; a paid OpenAI key can be used by
    setting `PROVIDER=openai`, `AI_BASE_URL=https://api.openai.com/v1`,
    and `MODEL=gpt-4o-mini`.
- **Schema/validation:** Pydantic v2.
- **Dependency management:** `uv` for local dev; a plain **`requirements.txt`**
  is committed so the grader installs with stock `pip`.

### Model & provider (exact versions)
| Setting  | Value |
|----------|-------|
| Provider | OpenRouter (free tier, OpenAI-compatible) |
| Base URL | `https://openrouter.ai/api/v1` |
| Model    | `meta-llama/llama-3.3-70b-instruct:free` |
| SDK      | `openai` 2.52.0 (pinned in `requirements.txt`), used verbatim |
| Python   | 3.12 |

---

## 2. Dataset

**BBC News articles** — the classic `bbc-text.csv` (category + full article
text), documented and widely mirrored on GitHub:

- Mirror: <https://github.com/susanli2016/NLP-with-Python/blob/master/data/bbc-text.csv>
- Raw: <https://raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/bbc-text.csv>
- Original academic source: D. Greene, P. Cunningham, *"Practical Solutions to
  the Problem of Diagonal Dominance in Kernel Document Clustering,"* ICML 2006.

The full CSV is **~2 MB**, so it is **not committed**; a small sample is
committed at `data/bbc-text-sample.csv` so the app runs out-of-the-box with no
downloads. To fetch the full file run:

```bash
uv run python scripts/download_dataset.py
# Saved dataset to ...\data\bbc-text.csv
```

---

## 3. Quick start (local, `uv`)

```bash
# 1. Create project (already has .env.example)
cp .env.example .env          # Windows:  Copy-Item .env.example .env

# 2. Create env + install pinned deps
uv venv --python 3.12
uv pip install -r requirements.txt

# 3. Run the FastAPI server
uv run uvicorn app.main:app --reload --port 8000
# -> http://localhost:8000/docs   (OpenAPI UI)
```

### Plain pip fallback (no uv installed)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate        # Unix
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

> **No API key required for OpenRouter free routing.** If you add a key, set
> `API_KEY` in `.env`. Never hardcode a key. `.env` is git-ignored.

---

## 4. Usage

### HTTP API

**Testing the endpoint with the interactive docs (recruiter/judge tip):** the
easiest way to try `/summarize` is the interactive OpenAPI page. With the
server running, open in your browser:

```
http://localhost:8000/docs#/
```

There you can expand `POST /summarize`, click **Try it out**, paste your text
in the `text` field, and press **Execute** to call the API from the browser —
no `curl` needed. The `/docs#/summarization/summarize_summarize_post` deep-link
just relaxes that same page on that endpoint.

You can also call it from the terminal:

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "The Bank of England held interest rates steady at 5.25% citing slowing inflation and a cooling labour market."}'
```

**Response**
```json
{
  "summary": "The Bank of England kept interest rates at 5.25%, pointing to easing inflation and a cooling jobs market as reasons for the hold.",
  "model": "meta-llama/llama-3.3-70b-instruct:free",
  "provider": "openai-compatible",
  "cached": false
}
```

Other endpoints:
- `GET /health` — status + active model.
- `GET /docs` / `GET /docs#/` — interactive OpenAPI documentation (use it to
  test the endpoint from the browser).

### CLI
Summarise a few articles straight from the terminal (same cached provider path):

```bash
uv run python -m app.cli --csv data/bbc-text-sample.csv -n 3
```

---

## 5. Docker (Bonus A)

Runs in a single container exposing the API on port 8000.

```bash
docker compose up --build
# or plain docker:
docker build -t bbc-summarizer .
docker run -p 8000:8000 -e API_KEY=... bbc-summarizer
```

Docker reads runtime settings from your `.env` (copy from `.env.example`).
A non-root user and a `HEALTHCHECK` are configured for security and
observability.

---

## 6. Caching + rate limiting (Bonus B)

- **Caching:** identical input text is served from
  an in-memory LRU cache with a configurable TTL (`CACHE_TTL_SECONDS`, default
  1 h). A cache hit returns `"cached": true` and never calls the API again.
- **Rate limiting:** a sliding window (`RATE_LIMIT_REQUESTS` per
  `RATE_LIMIT_WINDOW_SECONDS`) per client IP protects the endpoint from abuse
  and from exhausting the upstream provider's quota. Exceeding it returns
  `429 Too Many Requests`.

---

## 7. Configuration

All knobs live in `.env` (see `.env.example`). Read by a single Pydantic
`Settings` class (`app/config.py`) — never hardcoded.

| Variable | Default | Meaning |
|----------|---------|---------|
| `PROVIDER` | `openrouter` | Which provider the factory builds (`openrouter` or `openai`). |
| `AI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL. |
| `MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Model id. |
| `API_KEY` | *(empty)* | Secret key (env only). |
| `AI_TIMEOUT_SECONDS` | `30` | Per-request timeout. |
| `SUMMARY_MAX_TOKENS` | `180` | Max output tokens. |
| `SUMMARY_TEMPERATURE` | `0.4` | Sampling temperature. |
| `INPUT_MAX_WORDS` | `1500` | Rejects oversized inputs (> limit → 422). |
| `TENACITY_RETRIES` | `3` | Retries with exponential backoff. |
| `CACHE_ENABLED` / `CACHE_TTL_SECONDS` | `true` / `3600` | Cache toggle + TTL. |
| `RATE_LIMIT_ENABLED` / `REQUESTS` / `WINDOW` | `true` / `10` / `60` | Rate-limit toggle + window. |

---

## 8. Architecture

```
repo/
├── app/
│   ├── main.py                 # FastAPI wiring + routes (DI via Depends)
│   ├── config.py               # pydantic-settings Settings (env-driven)
│   ├── dataset.py             # Repository/adapter over the CSV
│   ├── cli.py                  # CLI entry point
│   ├── models/schemas.py       # Pydantic request/response models
│   ├── services/
│   │   ├── base_provider.py    # abstract AIProvider (Strategy)
│   │   ├── openai_provider.py  # OpenAI-compatible provider (retries, mapping)
│   │   └── provider_factory.py # Factory pattern
│   └── core/
│       ├── cache.py            # thread-safe TTL/LRU cache (Bonus B)
│       ├── rate_limiter.py     # sliding-window limiter (Bonus B)
│       └── exceptions.py       # domain errors -> HTTP status + exit codes
├── scripts/download_dataset.py # dataset downloader
├── data/bbc-text-sample.csv    # committed sample (git-ignored full CSV)
├── tests/test_api.py            # unit tests (provider mocked, no live calls)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt            # pinned deps (source of truth for grader)
└── README.md
```

**Design patterns (per the challenge brief):** Provider *(Strategy)*, provider
factory *(Factory)*, dependency injection via `Depends()` *(+/CLI DI)*, dataset
repository/adapter, caching & rate limiting as wrapping services *(Decorator
style)*, and tenacity retry with exponential backoff *(Resilience)*. Every
external-call failure (timeout, non-200, rate limit, malformed response, empty
text) maps to a typed exception and then to the right HTTP status code / CLI
exit code.

---

## 9. Tests

Unit tests mock the provider — **no live API calls**.

```bash
uv run pytest -q
```

Coverage includes: cache round-trip / TTL / eviction / invalidation, rate-limit
admission & window reset, provider error mapping (empty text, rate limit,
malformed response, connection error), and end-to-end `/summarize`, `/health`,
and cache-hit behavior via FastAPI `TestClient`.

---

## 10. Exit codes (CLI)

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Generic / dataset not found |
| 2 | Invalid input |
| 3 | Provider unavailable / timeout / network |
| 4 | Provider rate limited / auth |
| 5 | Malformed provider response |
| 6 | Local rate limit reached |

---

Built for *Python Run, Debug the Future — Challenge 2 (We Are Community)*.
All code, comments, and docs are in English.