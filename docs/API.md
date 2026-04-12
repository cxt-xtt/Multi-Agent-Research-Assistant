# API Reference — Multi-Agent Research Assistant

Base URL: `http://localhost:8000`  
Interactive Docs: `http://localhost:8000/docs`

---

## Authentication

All `/api/*` endpoints require an `X-API-Key` header matching the `SECRET_KEY` environment variable.

```
X-API-Key: your-secret-key
```

---

## Endpoints

### `POST /api/research/`

Runs the full multi-agent research pipeline for the given query.

**Pipeline stages:**
1. **SearchAgent** — fetches real-time web results via Tavily
2. **SummarizerAgent** — synthesizes results into a structured summary using GPT-4o
3. **FactCheckerAgent** — cross-references claims and assigns confidence scores

#### Request Body

```json
{
  "query": "What are the latest advances in quantum computing?",
  "search_depth": "advanced",
  "use_cache": true,
  "notify_n8n": true
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | ✅ | — | Research question (3–1000 chars) |
| `search_depth` | `"basic"` \| `"advanced"` | ❌ | `"advanced"` | Tavily search depth |
| `use_cache` | boolean | ❌ | `true` | Return cached result if available |
| `notify_n8n` | boolean | ❌ | `true` | Fire n8n webhook on completion |

#### Response `200 OK`

```json
{
  "query": "What are the latest advances in quantum computing?",
  "status": "completed",
  "started_at": "2024-01-15T10:00:00+00:00",
  "completed_at": "2024-01-15T10:00:14+00:00",
  "total_latency_ms": 14200.0,
  "node_timings_ms": {
    "search": 2100,
    "summarize": 6800,
    "fact_check": 5100
  },
  "errors": [],
  "direct_answer": "Quantum computers have achieved...",
  "sources": [
    {
      "title": "MIT Quantum Computing Breakthrough",
      "url": "https://news.mit.edu/quantum",
      "content": "Researchers have achieved...",
      "score": 0.94,
      "published_date": "2024-01-10"
    }
  ],
  "summary": {
    "summary": "Quantum computing has seen remarkable progress...",
    "key_points": [
      "IBM achieved 1000+ qubit processors in 2023",
      "Error correction rates improved by 40%"
    ],
    "key_entities": {
      "people": ["John Preskill"],
      "organizations": ["IBM", "MIT", "Google"],
      "dates": ["2023", "Q4 2024"],
      "statistics": ["1000 qubits", "40% improvement"]
    },
    "sources_used": [1, 2, 3],
    "confidence": 0.87
  },
  "fact_check": {
    "overall_confidence": 0.84,
    "verdict": "VERIFIED",
    "verdict_emoji": "✅",
    "claims": [
      {
        "claim": "IBM achieved 1000+ qubit processors",
        "confidence": 0.95,
        "status": "SUPPORTED",
        "supporting_sources": [1, 2],
        "note": "Confirmed by multiple independent sources"
      }
    ],
    "contradictions": [],
    "unverified_claims": [],
    "fact_check_summary": "All major claims are well-supported by sources."
  },
  "overall_confidence": 0.855,
  "_from_cache": false
}
```

#### Verdict Values

| Verdict | Emoji | Meaning |
|---|---|---|
| `VERIFIED` | ✅ | All major claims supported by multiple sources |
| `MOSTLY_VERIFIED` | 🟡 | Most claims supported; minor gaps exist |
| `UNCERTAIN` | ⚠️ | Insufficient source coverage to verify claims |
| `DISPUTED` | ❌ | Contradictions found across sources |

#### Error Responses

| Status | Description |
|---|---|
| `422` | Validation error (query too short, invalid depth) |
| `500` | Internal pipeline failure |

---

### `POST /api/research/feedback`

Logs user feedback for a pipeline run to LangSmith for quality tracking.

#### Request Body

```json
{
  "run_id": "lsm-run-abc123",
  "key": "accuracy",
  "score": 0.9,
  "comment": "Very accurate and well-sourced."
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `run_id` | string | ✅ | LangSmith run ID from pipeline execution |
| `key` | string | ✅ | Feedback dimension (`"accuracy"`, `"helpfulness"`, etc.) |
| `score` | float | ✅ | Score between `0.0` and `1.0` |
| `comment` | string | ❌ | Optional free-text comment |

#### Response

- `204 No Content` — Feedback logged successfully
- `503 Service Unavailable` — LangSmith is unreachable

---

### `GET /health`

Returns service health status and dependency connectivity.

#### Response `200 OK`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "openai": "configured",
    "tavily": "configured",
    "langsmith": "enabled",
    "redis": "configured",
    "n8n": "configured"
  },
  "uptime_seconds": 3601.4
}
```

---

## n8n Webhook

When `notify_n8n: true` (default), the pipeline POSTs the completed research report to:

```
POST $N8N_WEBHOOK_URL
Content-Type: application/json
X-Webhook-Secret: $N8N_WEBHOOK_SECRET
```

The n8n workflow (`n8n/workflow.json`) receives this payload, formats a Markdown report, and can route it to Slack, email, Notion, Google Docs, or any other integration.

---

## LangSmith Tracing

When `LANGCHAIN_TRACING_V2=true`, every pipeline run is traced with:

- Full agent execution tree with parent/child relationships
- Per-node token usage (prompt + completion tokens)
- Latency breakdown by node
- All intermediate inputs and outputs
- Error traces with stack context

View traces at: `https://smith.langchain.com/o/default/projects/p/{LANGCHAIN_PROJECT}`

---

## Error Handling

All errors follow this structure:

```json
{
  "detail": "Human-readable error message",
  "error_code": "PIPELINE_ERROR"
}
```

Pipeline errors are non-fatal where possible. If the SearchAgent fails, the pipeline returns a `partial` status with whatever results were collected, rather than failing entirely.