from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel
from app.models.entities import SearchResult

class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float

class IngestResponse(BaseModel):
    message: str
    directory_path: str
    files_processed: int
    files_skipped: int
    failed_files: int
    total_chunks: int
    processing_time_ms: float
    status: str

class CreateCollectionResponse(BaseModel):
    message: str
    collection_name: str
    created: bool
    status: str

class DeleteCollectionResponse(BaseModel):
    message: str
    collection_name: str
    deleted: bool
    status: str

class HybridSearchResult(BaseModel):
    chunk_id: str
    content: str
    file_name: str
    file_path: str
    file_type: str
    chunk_index: int
    semantic_score: float
    keyword_score: float
    exact_match_score: float
    final_score: float
    metadata: Dict[str, Any] = {}

class HybridSearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float
    search_breakdown: Dict[str, int] = {}

class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collection_exists: bool
    collection_name: str
    embedding_model: str
    uptime_seconds: Optional[float] = None
