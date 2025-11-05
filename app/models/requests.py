from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query string to search for.", min_length=1)
    top_k: int = Field(5, description="Number of top results to return.", ge=1, le=100)
    score_threshold: float = Field(0.0, description="Minimum score threshold for results.", ge=0.0, le=1.0)
    file_type: Optional[Union[ str, List[str]]] = Field(None, description="Filter results by file extensions (e.g., 'pdf', 'txt').")
    file_name: Optional[Union[str, List[str]]] = Field(None, description="Filter results by file name substring")

class IngestRequest(BaseModel):
    directory_path: str = Field(..., description="Path to the directory to ingest files from.", min_length=1)
    skip_existing: bool = Field(True, description="Whether to skip files that are already indexed.")
    chunk_size: int = Field(400, description="Chunk size for document chunking.", ge=1, le=512)
    overlap: int = Field(100, description="Overlap size for document chunking.", ge=0, le=511)
