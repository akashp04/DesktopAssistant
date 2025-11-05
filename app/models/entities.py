from typing import Dict, Any
from pydantic import BaseModel

class SearchResult(BaseModel):
    chunk_id: str
    file_name: str
    file_path: str
    file_type: str
    chunk_text: str
    chunk_index: int
    total_chunks: int
    score: float
    metadata: Dict[str, Any]