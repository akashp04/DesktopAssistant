from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query string to search for.", min_length=1)
    top_k: int = Field(5, description="Number of top results to return.", ge=1, le=100)
    score_threshold: float = Field(0.0, description="Minimum score threshold for results.", ge=0.0, le=1.0)
    file_type: Optional[Union[ str, List[str]]] = Field(None, description="Filter results by file extensions (e.g., 'pdf', 'txt').")
    file_name: Optional[Union[str, List[str]]] = Field(None, description="Filter results by file name substring")

class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="The search query", min_length=1)
    top_k: int = Field(5, description="Number of results to return", ge=1, le=100)
    semantic_weight: float = Field(1.0, description="Weight for semantic search", ge=0.0, le=10.0)
    keyword_weight: float = Field(1.0, description="Weight for keyword search", ge=0.0, le=10.0)
    exact_weight: float = Field(2.0, description="Weight for exact matches", ge=0.0, le=10.0)
    file_type: Optional[Union[str, List[str]]] = Field(None, description="Filter by file type")
    file_name: Optional[Union[str, List[str]]] = Field(None, description="Filter by file name")

class IngestRequest(BaseModel):
    directory_path: str = Field(..., description="Path to the directory to ingest files from.", min_length=1)
    skip_existing: bool = Field(True, description="Whether to skip files that are already indexed.")
    chunk_size: int = Field(400, description="Chunk size for document chunking.", ge=1, le=512)
    overlap: int = Field(100, description="Overlap size for document chunking.", ge=0, le=511)
    max_workers: int = Field(3, description="Number of parallel threads", ge=1, le=8)

class WatchFolderRequest(BaseModel):
    folder_path: str
    allowed_extensions: Optional[List[str]] = None
    recursive: bool = True
    auto_ingest: bool = True