# Desktop Assistant

An intelligent document management system with Retrieval-Augmented Generation (RAG) capabilities, featuring hybrid search, Context Enriched Chunking, and real-time folder monitoring. Upload documents, perform semantic search with keyword fusion, and get relevant answers from your document corpus with automatic updates.

## Features
- **Hybrid Search**: Combines semantic similarity, BM25 keyword search, and exact matching with Reciprocal Rank Fusion (RRF)
- **Context Enriched Chunking**: Semantic-aware chunking that respects sentence boundaries and enriches metadata
- **Real-time Folder Monitoring**: Automatic document ingestion with background file system watching
- **Document Ingestion**: Supports PDF, DOC/DOCX, PPT/PPTX, TXT, MD, RTF, and ODT files
- **Semantic Search**: Vector-based search using Qdrant vector database
- **Text Extraction**: Powered by the `unstructured` library
- **Smart Chunking**: Token-aware chunking with semantic boundaries and metadata enrichment
- **Embeddings**: Uses SentenceTransformers for high-quality embeddings
- **FastAPI**: REST API with automatic documentation and comprehensive endpoints
- **Flexible Storage**: Supports both local file storage and Qdrant server modes

## Project Structure
```
DesktopAssistant/
├── app/                    # FastAPI application
│   ├── api/               # API endpoints
│   │   ├── search.py      # Search and ingestion endpoints
│   │   ├── collections.py # Collection management
│   │   └── watcher.py     # Folder watcher endpoints
│   ├── core/              # Core business logic
│   │   └── query_service.py # Central orchestration service
│   ├── models/            # Pydantic models
│   │   ├── requests.py    # API request models
│   │   └── responses.py   # API response models
│   └── config.py          # Configuration settings
├── storage/               # Storage and retrieval components
│   ├── vector_storage.py  # Qdrant integration
│   ├── embedding_manager.py # Embedding generation
│   ├── document_chunker.py  # Context enriched chunking
│   ├── file_handler.py     # File processing
│   ├── hybrid_searcher.py  # Hybrid search implementation
│   └── pipeline.py        # Ingestion pipeline
├── watchservice/          # Folder monitoring service
│   ├── watcher.py         # File system watcher
│   └── watcherservice.py  # Watcher service management
├── static/                # Frontend assets
├── qdrant_storage/        # Local Qdrant data (if using path mode)
└── requirements.txt       # Dependencies
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
pip install -r requirements.txt
```

## Configuration
Settings are defined in `app/config.py` using `pydantic-settings`. You can override via environment variables or a `.env` file in the project root.

Key settings (defaults shown):
- **APP**: `APP_NAME=Desktop Assistant`, `APP_VERSION=1.0.0`
- **API**: `HOST=localhost`, `PORT=8000`, `RELOAD=true`, `LOG_LEVEL=info`
- **Qdrant**: `QDRANT_HOST=localhost`, `QDRANT_PORT=6333`, `QDRANT_PATH=./qdrant_storage`, `COLLECTION_NAME=documents`
- **Embeddings**: `EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5`, `VECTOR_SIZE=384`
- **Chunking**: `CHUNK_SIZE=400`, `CHUNK_OVERLAP=100`, `MIN_CHUNK_SIZE=50`
- **Search**: `TOP_K=5`, `MAX_TOP_K=100`, `SCORE_THRESHOLD=0.0`
- **Hybrid Search**: `SEMANTIC_WEIGHT=0.7`, `KEYWORD_WEIGHT=0.3`, `RRF_K=60`
- **Folder Watcher**: `WATCHER_BATCH_SIZE=10`, `WATCHER_POLL_INTERVAL=2.0`

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

### Core Endpoints
- **GET** `/` – Basic health message
- **GET** `/health` – Verifies Qdrant connection and collection presence

### Search & Ingestion
- **POST** `/search` – Hybrid semantic search with BM25 and exact matching
- **POST** `/ingest` – Ingest a directory of files with Context Enriched Chunking

### Collections Management
- **GET** `/collections/info` – Info about current collection
- **POST** `/collections/{collection_name}/create` – Create a collection
- **GET** `/collections/{collection_name}/info` – Info for a specific collection
- **DELETE** `/collections/{collection_name}` – Delete a collection

### Folder Watcher (Real-time Monitoring)
- **POST** `/watcher/start` – Start folder monitoring service
- **POST** `/watcher/stop` – Stop folder monitoring service
- **GET** `/watcher/status` – Get watcher service status
- **POST** `/watcher/folders/add` – Add folder to monitoring
- **POST** `/watcher/folders/remove` – Remove folder from monitoring
- **GET** `/watcher/folders` – List monitored folders
- **POST** `/watcher/folders/batch` – Batch add/remove folders
- **POST** `/watcher/scan` – Force scan of all monitored folders

### Request Schemas

#### QueryRequest (Hybrid Search)
- `query`: string (required)
- `top_k`: int (default 5, 1–100)
- `score_threshold`: float (default 0.0)
- `file_type`: string or list of strings (e.g. "pdf", "txt")
- `file_name`: string or list of strings
- `search_type`: string (default "hybrid", options: "hybrid", "semantic", "keyword")

#### IngestRequest (Context Enriched Chunking)
- `directory_path`: string (required)
- `skip_existing`: bool (default true)
- `chunk_size`: int (default 400, ≤512)
- `overlap`: int (default 100, < chunk_size)
- `min_chunk_size`: int (default 50)

#### WatcherFolderRequest
- `folder_path`: string (required)

#### WatcherBatchRequest
- `add_folders`: list of strings (optional)
- `remove_folders`: list of strings (optional)

### Usage Examples

#### Ingest Documents with Context Enriched Chunking
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "/DesktopAssistant/Books",
    "skip_existing": true,
    "chunk_size": 400,
    "overlap": 100,
    "min_chunk_size": 50
  }'
```

#### Hybrid Search (Semantic + Keyword + Exact)
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what happened in chamber of secrets?",
    "top_k": 5,
    "score_threshold": 0.0,
    "file_type": ["pdf", "txt"],
    "file_name": "Harry Potter",
    "search_type": "hybrid"
  }'
```

#### Start Folder Monitoring
```bash
# Start the watcher service
curl -X POST "http://localhost:8000/watcher/start"

# Add a folder to monitor
curl -X POST "http://localhost:8000/watcher/folders/add" \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/DesktopAssistant/Books"}'

# Check watcher status
curl -X GET "http://localhost:8000/watcher/status"
```

#### Batch Folder Management
```bash
curl -X POST "http://localhost:8000/watcher/folders/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "add_folders": ["/path/to/docs", "/path/to/papers"],
    "remove_folders": ["/old/path"]
  }'
```

## Hybrid Search & Re-ranking ✅

The system features a sophisticated hybrid search that combines multiple retrieval methods:

- **Semantic Similarity**: Vector search in Qdrant using embeddings
- **Keyword Scoring**: BM25 algorithm for term-based matching
- **Exact Match Heuristics**: Boosted scoring for exact phrase matches
- **Reciprocal Rank Fusion (RRF)**: Intelligent fusion of results with configurable weights

### Search Types
1. **Hybrid Search** (default): Combines all methods with RRF fusion
2. **Semantic Search**: Pure vector similarity search
3. **Keyword Search**: BM25-based term matching

### Configuration
Adjust hybrid search behavior via environment variables:
```bash
SEMANTIC_WEIGHT=0.7    # Weight for semantic similarity
KEYWORD_WEIGHT=0.3     # Weight for keyword matching
RRF_K=60              # RRF parameter for fusion smoothing
```

### Programmatic Usage
```python
from app.core.query_service import QueryService

query_service = QueryService()
results = query_service.hybrid_search(
    query="chamber of secrets",
    top_k=5,
    search_type="hybrid"
)
```

## Context Enriched Chunking ✅

Advanced chunking strategy that preserves semantic coherence and enriches metadata:

### Features
- **Semantic Boundaries**: Respects sentence boundaries to avoid mid-sentence splits
- **Token-Aware Sizing**: Uses `tiktoken` for accurate token counting
- **Metadata Enrichment**: Adds comprehensive context including:
  - Document structure (sections, paragraphs)
  - Chunk position and relationships
  - Content type classification
  - Quality metrics and statistics

### Chunking Process
1. **Text Extraction**: Uses `unstructured` library for robust document parsing
2. **Semantic Segmentation**: Identifies natural break points (sentences, paragraphs)
3. **Intelligent Sizing**: Balances chunk size with semantic coherence
4. **Context Preservation**: Maintains overlap while respecting boundaries
5. **Metadata Enhancement**: Enriches each chunk with structural and contextual information

## Real-Time Folder Monitoring ✅

Automatic document ingestion with background file system monitoring:

### Features
- **Real-Time Detection**: Monitors file system changes (create, modify, delete)
- **Batch Processing**: Groups file events for efficient processing
- **Background Operation**: Non-blocking service that runs independently
- **Selective Monitoring**: Add/remove specific folders dynamically
- **Force Scanning**: On-demand scanning of monitored folders

### Watcher Service
The folder watcher automatically:
1. Detects new or modified files in monitored directories
2. Processes files using the same Context Enriched Chunking pipeline
3. Updates the vector database with new embeddings
4. Removes vectors for deleted files
5. Provides status updates via API endpoints

## Ingestion Pipeline

### Automated Process
- Discovers files via `file_handler.py` (supports: .pdf, .docx, .doc, .pptx, .ppt, .txt, .md, .rtf, .odt)
- Extracts text using `unstructured` with enhanced parsing
- Applies Context Enriched Chunking with `document_chunker.py`
- Generates embeddings with `embedding_manager.py` (SentenceTransformers)
- Stores vectors and enriched payloads to Qdrant with `vector_storage.py`
- Builds BM25 keyword index for hybrid search capabilities

### Manual Pipeline Execution
```bash
python /DesktopAssistant/storage/pipeline.py \
  --base_directory "/DesktopAssistant/Books" \
  --embedding_model "BAAI/bge-small-en-v1.5" \
  --qdrant_path "/DesktopAssistant/qdrant_storage" \
  --chunk_size 400 \
  --overlap 100 \
  --min_chunk_size 50 \
  --skip_existing
```

## System Architecture

### Core Components
- **QueryService**: Central orchestration service with lazy loading pattern
- **HybridSearch**: Multi-modal retrieval with RRF fusion
- **VectorStorage**: Qdrant integration with collection management
- **EmbeddingManager**: SentenceTransformers-based embedding generation
- **FolderWatcher**: Real-time file system monitoring service
- **DocumentChunker**: Context-aware chunking with semantic boundaries

### Design Patterns
- **Dependency Injection**: Clean separation of concerns
- **Lazy Loading**: Circular dependency resolution
- **Event-Driven**: Asynchronous file processing
- **Microservice-Ready**: Modular API structure

### Performance Features
- **Embedded Qdrant**: Local storage mode for zero-setup deployment
- **Batch Processing**: Efficient handling of multiple file operations
- **Background Services**: Non-blocking folder monitoring
- **Configurable Weights**: Tunable hybrid search parameters

## Future Enhancements

### Multi-Agent Orchestration
Planning for intelligent document processing with specialized agents:
- **Document Classifier**: Automatic content categorization
- **Metadata Extractor**: Enhanced document analysis
- **Query Router**: Intent-based search routing
- **Response Synthesizer**: Context-aware answer generation

### TODO
- [ ] Multi-agent architecture implementation
- [ ] Advanced document understanding (tables, images, charts)
- [ ] Conversation memory and context tracking
- [ ] Custom embedding fine-tuning pipeline
- [ ] Distributed deployment support
- [ ] Complete Desktop App using Tauri or Electron

