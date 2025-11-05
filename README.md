# DesktopAssistant

A local-first document assistant that indexes your files into a vector database (Qdrant) and exposes a FastAPI service for semantic search and administration. It supports ingestion from a directory of mixed file types, chunking, embedding via SentenceTransformers, storage in Qdrant, and querying with optional filters.

## Features
- Semantic embeddings with `sentence-transformers` (default: `BAAI/bge-small-en-v1.5`)
- Chunking with token-aware sizes using `tiktoken`
- File parsing via `unstructured`
- Qdrant vector store with automatic collection management
- FastAPI endpoints for health, search, ingestion, and collection admin
- Hybrid search (semantic + BM25 + exact match) with Reciprocal Rank Fusion re-ranking

## Project Structure
```
DesktopAssistant/
  app/
    app.py                 # FastAPI app and endpoints
    config.py              # Pydantic settings (env-driven)
    core/query_service.py  # Orchestration for search and ingestion
    models/
      entities.py          # Pydantic entities (SearchResult)
      requests.py          # Request schemas (QueryRequest, IngestRequest)
      responses.py         # Response schemas
  storage/
    document_chunker.py    # Token-aware chunking
    embedding_manager.py   # SentenceTransformer wrapper
    file_handler.py        # File discovery and text extraction
    pipeline.py            # Parallel ingestion pipeline
    vector_storage.py      # Qdrant integration
    hybrid_searcher.py     # Hybrid search utilities (BM25, exact, RRF)
  Books/                   # Example documents
  qdrant_storage/          # Local Qdrant data (if using path-based client)
  .gitignore
  README.md
```

## Requirements
- Python 3.10+
- Qdrant (either running as a service on `localhost:6333` or local file-path mode)

Python dependencies (install via pip):
- fastapi, uvicorn
- pydantic, pydantic-settings
- qdrant-client
- sentence-transformers, numpy
- tiktoken
- unstructured

Example installation:
```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn pydantic pydantic-settings qdrant-client sentence-transformers numpy tiktoken unstructured
```

## Configuration
Settings are defined in `app/config.py` using `pydantic-settings`. You can override via environment variables or a `.env` file in the project root.

Key settings (defaults shown):
- APP: `APP_NAME=Desktop Assistant`, `APP_VERSION=1.0.0`
- API: `HOST=localhost`, `PORT=8000`, `RELOAD=true`, `LOG_LEVEL=info`
- Qdrant: `QDRANT_HOST=localhost`, `QDRANT_PORT=6333`, `QDRANT_PATH=None`, `COLLECTION_NAME=documents`
- Embeddings: `EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5`, `VECTOR_SIZE=384`
- Chunking: `CHUNK_SIZE=400`, `CHUNK_OVERLAP=100`
- Search: `TOP_K=5`, `MAX_TOP_K=100`, `SCORE_THRESHOLD=0.0`

Notes:
- If `QDRANT_PATH` is set, the client will use local file storage (and ignore host/port). Ensure the directory exists or can be created by the process. The repository already includes a `qdrant_storage/` directory if you prefer path mode.
- `VECTOR_SIZE` must match the embedding model’s output dimension.

## Running the API
Start the FastAPI server (two options):

1) Using uvicorn import string
```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

2) Using the module directly
```bash
python /DesktopAssistant/app/app.py
```

Open the interactive docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Qdrant Setup
You can use either:
- Service mode: Run Qdrant locally (e.g., Docker) and use `QDRANT_HOST`/`QDRANT_PORT`.
- Path mode: Set `QDRANT_PATH=/DesktopAssistant/qdrant_storage` to store data locally without a running service.

On first run, `vector_storage.py` will create the collection if it does not exist.

## API Endpoints
- GET `/` – Basic health message
- GET `/health` – Verifies Qdrant connection and collection presence
- POST `/search` – Semantic search
- POST `/ingest` – Ingest a directory of files
- GET `/collections/info` – Info about current collection
- POST `/collections/{collection_name}/create` – Create a collection
- GET `/collections/{collection_name}/info` – Info for a specific collection
- DELETE `/collections/{collection_name}` – Delete a collection

### Schemas
- QueryRequest
  - `query`: string (required)
  - `top_k`: int (default 5, 1–100)
  - `score_threshold`: float (default 0.0)
  - `file_type`: string or list of strings (e.g. "pdf", "txt")
  - `file_name`: string or list of strings

- IngestRequest
  - `directory_path`: string (required)
  - `skip_existing`: bool (default true)
  - `chunk_size`: int (default 400, ≤512)
  - `overlap`: int (default 100, < chunk_size)

### Example: Ingest
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "/DesktopAssistant/Books",
    "skip_existing": true,
    "chunk_size": 400,
    "overlap": 100
  }'
```

### Example: Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what happened in chamber of secrets?",
    "top_k": 5,
    "score_threshold": 0.0,
    "file_type": ["pdf", "txt"],
    "file_name": "Harry Potter"
  }'
```

## Hybrid Search + Re-ranking
This repo includes a hybrid search module that combines:
- Semantic similarity (vector search in Qdrant)
- Keyword scoring (BM25)
- Exact match heuristics

The results are merged with Reciprocal Rank Fusion (RRF) for stronger retrieval quality. It is implemented in `storage/hybrid_searcher.py` and not wired into the HTTP API by default.

Minimal usage example:
```python
from DesktopAssistant.storage.hybrid_searcher import BM25Search, ExactMatchSearch, RRF, HybridSearch

# Reuse existing instances if you have a running app
from DesktopAssistant.storage.vector_storage import VectorStorage
from DesktopAssistant.storage.embedding_manager import EmbeddingManager

vector_storage = VectorStorage(collection_name="documents")
embedding_manager = EmbeddingManager("BAAI/bge-small-en-v1.5")

hybrid = HybridSearch(
    vector_storage=vector_storage,
    embedding_manager=embedding_manager,
    bm25=BM25Search(),
    exact_matcher=ExactMatchSearch(),
    rrf_ranker=RRF(k=60)
)

hybrid.build_keyword_index()
results = hybrid.hybrid_search(query="chamber of secrets", top_k=5)
```

To integrate into the API, you can instantiate `HybridSearch` alongside `QueryService`, build the keyword index on startup, and call `hybrid.hybrid_search` inside the search handler. Map the combined results back to your `SearchResult` model.

## Ingestion Pipeline
- Discovers files in a directory via `file_handler.py` (allowed: .pdf, .docx, .doc, .pptx, .ppt, .txt, .md, .rtf, .odt)
- Extracts text using `unstructured`
- Chunks with `document_chunker.py` (token-aware via `tiktoken`)
- Generates embeddings with `embedding_manager.py` (SentenceTransformers)
- Stores vectors and payloads to Qdrant with `vector_storage.py`

You can also run the pipeline directly:
```bash
python /DesktopAssistant/storage/pipeline.py \
  --base_directory "/DesktopAssistant/Books" \
  --embedding_model "BAAI/bge-small-en-v1.5" \
  --qdrant_path "/DesktopAssistant/qdrant_storage" \
  --chunk_size 400 \
  --overlap 100 \
  --skip_existing
```

## Desktop App Bundling
When all the functionalites are completed, I'll ship it as a Desktop App 
- Full desktop UI (webview): use Tauri or Electron.
```
