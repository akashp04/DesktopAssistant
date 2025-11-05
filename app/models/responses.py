from typing import List, Optional, Union
from pydantic import BaseModel
from DesktopAssistant.app.models.entities import SearchResult

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
