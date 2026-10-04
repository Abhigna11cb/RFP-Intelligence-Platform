
# RFP Intelligence Platform

> AI-powered RAG Search Engine & Multi-Agent System for bid/RFP document analysis.

---

## Architecture

```
HTML / PDF files (Bid1, Bid2, ...)
         │
         ▼
┌─────────────────────────────────┐
│   Ingestion & Parsing           │
│   pymupdf4llm · BeautifulSoup   │
│   OCR (Tesseract) · tiktoken    │
└──────────────┬──────────────────┘
               │  chunks + metadata
               ▼
┌─────────────────────────────────┐
│   PostgreSQL 15 (pgvector)      │
│   ├── vector(3072) exact scan   │  ← text-embedding-3-large (full dims)
│   └── tsvector GIN idx          │  ← BM25 / full-text search
└──────────────┬──────────────────┘
               │  search tool
               ▼
┌──────────────────────────────────────────────────────┐
│   LangGraph Multi-Agent Orchestrator                 │
│                                                      │
│  ┌─────────────┐    ┌──────────────┐                 │
│  │ Orchestrator│───▶│  Retrieval   │ hybrid_search   │
│  │   (Planner) │    │    Agent     │ + reranker      │
│  └─────────────┘    └──────┬───────┘                 │
│         │                  │ evidence                │
│         ▼                  ▼                         │
│  ┌─────────────┐    ┌──────────────┐                 │
│  │  Addendum   │◀───│  Extraction  │ 20 fields       │
│  │Reconciliation│   │   Agent(s)   │ + citations     │
│  └──────┬──────┘    └──────────────┘                 │
│         │ updated fields                             │
│         ▼                                            │
│  ┌─────────────┐    ┌──────────────┐                 │
│  │  Validator  │───▶│  Q&A / Report│ JSON + answer   │
│  │   /Critic   │    │    Agent     │ with citations  │
│  └─────────────┘    └──────────────┘                 │
└──────────────────────────────────────────────────────┘
         │
         ▼
  FastAPI REST API  /  Streamlit Web UI  /  CLI (main.py)
```

---

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL **15** with [pgvector](https://github.com/pgvector/pgvector) extension
- OpenAI API key (for embeddings + LLM)

---

### Step 1 — Create a virtual environment

```bash
# Windows (PowerShell)
cd "Assignment-Data-Statements (AI Engineer-Emplay Inc)"
python -m venv venv
.\venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First run downloads the cross-encoder model (~85 MB) — requires internet.

---

### Step 3 — Configure environment variables

```bash
# Copy the example file
copy .env.example .env       # Windows
cp .env.example .env         # Linux/Mac
```

Then open `.env` and fill in your values:

```env
# PostgreSQL 15
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rfp_platform
DB_USER=postgres
DB_PASSWORD=your_postgres_password

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
EMBEDDING_DIMENSIONS=3072
EMBEDDING_PROVIDER=openai
```

---

### Step 4 — Create the database

```bash
# Creates the rfp_platform database, enables pgvector, and creates all tables
python scratch/setup_pg15.py
```

> ℹ️ Requires PostgreSQL 15 to be running locally on port 5432.

---

### Step 5 — Index bid documents

```bash
# Index both provided bids (run once — incremental, safe to re-run)
python main.py index --bid ./Bid1 --bid ./Bid2
```

This parses all PDFs/HTML files, chunks them, embeds via `text-embedding-3-large`, and stores in pgvector. Takes ~2–3 minutes.

---

## Run the Frontend (Streamlit Web UI)

```bash
# Make sure the venv is activated first
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux/Mac

# Launch the UI
streamlit run app_ui.py
```

Then open your browser at:

**→ [http://localhost:8501](http://localhost:8501)**

### What you can do in the UI

| Tab | Description |
|-----|-------------|
| 🔍 **Search** | Semantic + keyword hybrid search across all bid documents. Filter by doc type (RFP, addendum, specs). |
| 💬 **Q&A Chat** | Ask natural language questions — answers are cited with source file + page number. |
| 📊 **Extract Fields** | Run the full 20-field extraction pipeline for any bid. Download the JSON result.|
| 📈 **Evaluation** | Run the 17-question retrieval benchmark across 4 search configurations.|

### Example questions to try

```
What is the submission deadline for Bid1 after all addendums?
What laptop model and specs are required in Bid2?
Is a bid bond required for Bid1?
What cooperative contract does Dallas ISD use?
Which affidavits are required for the Dell laptop bid?
Compare the delivery requirements of both bids.
```

> 💡 Use the **sidebar** to switch between "All Bids", "Bid1", or "Bid2" scope.

---

## Running the System (CLI / API)

### CLI commands

```bash
# Extract all 20 fields for a bid
python main.py extract --bid Bid1 --output Bid1_output.json
python main.py extract --bid Bid2 --output Bid2_output.json

# Ask a question (prints cited answer to console)
python main.py ask "What is the final submission deadline for Bid1?"
python main.py ask "Which affidavits are required for Bid2?" --bid Bid2
python main.py ask "Compare the warranty requirements of both bids."

# Run retrieval evaluation (17 questions × 4 configs)
python main.py eval

# Start REST API server
python main.py serve
```

### REST API

```bash
python main.py serve
# Swagger docs at http://localhost:8000/docs
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/index` | Index a bid folder |
| `GET`  | `/bids` | List all indexed bids |
| `GET`  | `/bids/{bid_id}/documents` | List documents for a bid |
| `POST` | `/search` | Direct hybrid search |
| `POST` | `/extract` | Run extraction pipeline |
| `GET`  | `/results/{bid_id}` | Get saved extraction results |
| `POST` | `/ask` | Answer a free-form question |
| `GET`  | `/addendums/{bid_id}` | Get addendum content |

### Run tests

```bash
# All tests
python -m pytest rfp_platform/tests/ -v

# Unit tests only (no DB needed)
python -m pytest rfp_platform/tests/test_ingestion.py -v

# Integration tests (requires DB with indexed data)
python -m pytest rfp_platform/tests/test_search.py -v -m integration
```

---

## Design Decisions

### Embedding Model: `text-embedding-3-large` at full 3072-dim
- **Dimensions used:** 3072 (full native size — no truncation). The pgvector column is `vector(3072)`, matching both storage and query embeddings exactly.
- **Why not all-mpnet-base-v2?** OpenAI's model scores ~15% higher on MTEB retrieval benchmarks, especially for domain-specific legal/procurement text. At ~$0.02 for the full corpus, cost is negligible.
- **Why 3072 instead of truncating to 1536?** For our corpus size (~126 chunks), exact nearest-neighbour scan is sub-millisecond regardless of dimension. Full 3072-dim gives the best possible retrieval quality at no practical performance cost.

### Chunking Strategy: Heading-Aware + Table-Aware
- `CHUNK_MIN_CHARS=1000`, `CHUNK_MAX_CHARS=3000`, `CHUNK_OVERLAP=200`
- **Why?** RFP sections like "Submission Requirements" or "Specifications" are self-contained units. Heading-aware splitting keeps the section context with its content. Tables are kept as single chunks to avoid breaking row/column relationships.
- Overlap prevents information loss at boundaries (critical for date/value extraction).

### Vector Store: PostgreSQL 15 + pgvector
- **Why not FAISS/Chroma?** We already need a relational DB for metadata. pgvector gives vector search + BM25 + metadata filtering + ACID transactions in one system. Zero additional infrastructure.
- `ivfflat` index with `lists=50` — good for our corpus size (~126 chunks, expandable to millions).

### Hybrid Search: RRF (Reciprocal Rank Fusion)
- Merges dense vector results and BM25 results with `RRF_K=60`.
- **Why RRF over weighted scores?** RRF is parameter-free and robust — no per-query score calibration needed. Particularly important for RFPs where some queries are conceptual ("submission logistics") and others are exact ("JA-207652").

### Re-ranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Re-scores top-20 candidates from RRF with a cross-encoder.
- **Why?** Cross-encoders process (query, document) jointly — capturing fine-grained interaction signals that bi-encoders miss. 22M params means CPU-fast inference.

### Agent Framework: LangGraph
- **Why LangGraph over CrewAI/AutoGen?** LangGraph's StateGraph gives explicit control over agent routing, conditional edges (retry loops), and shared state — exactly what the assignment requires. CrewAI is higher-level but less controllable for the validation feedback loop.

### LLM: GPT-4o-mini
- Good JSON adherence at low cost (~$0.15/1M tokens). Swappable via `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=ollama` in `.env`.

---

## Retrieval Evaluation Results

17 questions across both bids (10 × Bid1, 7 × Bid2). See [`eval_results.json`](eval_results.json) for full per-question breakdown.

| Configuration | Recall@5 | Recall@10 | MRR |
|---------------|----------|-----------|-----|
| vector_only | 0.941 | 0.941 | 0.804 |
| keyword_only | 0.176 | 0.176 | 0.147 |
| hybrid (RRF) | 0.941 | 0.941 | 0.804 |
| **hybrid + rerank** | **0.941** | **0.941** | **0.835** |

> Run `python main.py eval` to generate `eval_results.json` with actual scores.

See also:
- [`qa_sample_log.json`](qa_sample_log.json) — 10+ cited Q&A examples
- [`agent_trace_example.json`](agent_trace_example.json) — full extraction agent trace

---

## Output Format

```json
{
  "bid_id": "Bid1",
  "fields": {
    "Due Date": {
      "value": "2024-07-09 14:00 CST",
      "sources": [{"file": "Addendum 2 RFP JA-207652...pdf", "page": 1, "text": "..."}],
      "confidence": 0.95,
      "notes": "Extended by Addendum 2 (original: 2024-06-27)"
    },
    "Bid Bond Requirement": {
      "value": null,
      "sources": [],
      "confidence": 0.80,
      "notes": "Not found in documents"
    }
  },
  "addendum_changes": [
    {
      "addendum_number": 2,
      "field_name": "Due Date",
      "original_value": "2024-06-27 14:00 CST",
      "new_value": "2024-07-09 14:00 CST",
      "notes": "Due date extended by Addendum 2"
    }
  ],
  "validation": {"passed": 18, "failed": 0, "not_found": 2},
  "run_id": "79e93e0b-..."
}
```

---

## Project Structure

```
rfp_platform/
├── core/
│   ├── config.py       # pydantic-settings, all env vars
│   └── database.py     # connection pool, schema DDL
├── ingestion/
│   ├── chunker.py      # heading/table-aware chunking with overlap
│   ├── pipeline.py     # async ingestion: parse → chunk → embed → store
│   └── cli.py          # CLI wrapper
├── search/
│   ├── embeddings.py   # OpenAI text-embedding-3-large provider
│   ├── tools.py        # 8 search tools (hybrid, semantic, keyword, etc.)
│   └── reranker.py     # cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2)
├── agents/
│   ├── state.py        # AgentState, FieldResult, Citation (Pydantic)
│   ├── llm_client.py   # OpenAI/Anthropic LLM + tool execution loop
│   ├── nodes.py        # 6 agent node functions
│   ├── orchestrator.py # LangGraph graph + run_extraction/run_qa
│   ├── tool_schemas.py # OpenAI function-calling schemas
│   └── observability.py# Agent step logging to agent_logs table
├── api/
│   └── main.py         # FastAPI REST API (8 endpoints)
├── eval/
│   └── evaluate.py     # 17-question benchmark, 4 configs
└── tests/
    ├── test_ingestion.py # Unit tests: doc_type, chunker, hash
    └── test_search.py    # Integration tests: search, reranker, state

main.py                  # Single CLI entry point
app_ui.py                # Streamlit Web UI (bonus)
requirements.txt
.env / .env.example
.gitignore
```

---

## Known Limitations & Assumptions

1. **Addendum detection** uses filename patterns — assumes addendums are named "Addendum N ..." (standard practice).
2. **Table extraction** uses markdown conversion via pymupdf4llm — complex nested tables may not parse perfectly.
3. **ivfflat index** requires ≥50 rows to be effective; small test corpora fall back to sequential scan automatically.
4. **Re-ranker** (ms-marco-MiniLM-L-6-v2) is downloaded on first use (~85MB), requires internet connection.
5. **Extraction accuracy** depends on GPT-4o-mini's ability to parse RFP-specific language — tested on both provided bids.

---

## Bonus Features Implemented

- ✅ **OCR support** (Tesseract) for scanned/image-based PDFs
- ✅ **Cross-encoder re-ranking** (ms-marco-MiniLM-L-6-v2)
- ✅ **Streamlit Web UI** with Search, Chat, Extract, and Eval tabs
- ✅ **Observability** — every agent step logged to `agent_logs` table (latency, tokens, tool calls)
- ✅ **Incremental indexing** — SHA-256 dedup prevents re-embedding existing files

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5432 | PostgreSQL port |
| `DB_NAME` | rfp_platform | Database name |
| `DB_USER` | postgres | Database user |
| `DB_PASSWORD` | — | Database password |
| `EMBEDDING_MODEL` | text-embedding-3-large | OpenAI embedding model |
| `EMBEDDING_DIM` | 3072 | Vector dimension — full native size of text-embedding-3-large |
| `EMBEDDING_DIMENSIONS` | 3072 | OpenAI API `dimensions` param — must match `EMBEDDING_DIM` |
| `EMBEDDING_PROVIDER` | openai | `openai` or `local` |
| `LLM_PROVIDER` | openai | `openai`, `anthropic`, `ollama` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | gpt-4o-mini | LLM model name |
| `CHUNK_MIN_CHARS` | 1000 | Min chunk size |
| `CHUNK_MAX_CHARS` | 3000 | Max chunk size |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `DEFAULT_TOP_K` | 10 | Default search results |
| `RRF_K` | 60 | RRF fusion constant |
=======

