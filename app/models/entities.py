from typing import Dict, Any, Optional, List
from pydantic import BaseModel

class SearchResult(BaseModel):
    chunk_id: str
    file_name: str
    file_path: str
    file_type: str
    chunk_text: str
    chunk_index: int
    total_chunks: int
    score: float = 0.0
    
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    exact_match_score: float = 0.0
    final_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = {}

    class Config:
        from_attributes = True

class WatchedFolderInfo(BaseModel):
    path: str
    allowed_extensions: List[str]
    recursive: bool
    auto_ingest: bool
    added_at: float