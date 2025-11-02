from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import List, Optional, Dict, Any, Union
import time

from embedding_manager import EmbeddingManager
from vector_storage import VectorStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Desktop Assistant API",
    description="API for managing and querying desktop assistant embeddings and vectors.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query string to search for.", min_length=1)
    top_k: int = Field(5, description="Number of top results to return.", ge=1, le=100)
    score_threshold: float = Field(0.0, description="Minimum score threshold for results.", ge=0.0, le=1.0)
    file_type: Optional[Union[ str, List[str]]] = Field(None, description="Filter results by file extensions (e.g., 'pdf', 'txt').")
    file_name: Optional[Union[str, List[str]]] = Field(None, description="Filter results by file name substring")

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

class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float

class QueryService:
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 qdrant_path: str = None,
                 collection_name: str = "documents",
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        
        self.embedding_manager = EmbeddingManager(embedding_model = embedding_model_name)
        self.vector_storage = VectorStorage(
            host=qdrant_host,
            port=qdrant_port,
            path=qdrant_path,
            collection_name=collection_name
        )
        logger.info ("Query Service initialized.")
    
    def search(self, request: QueryRequest) -> QueryResponse:
        start = time.monotonic()
        try:
            # query_embedding = self.embedding_manager.generate_embedding(request.query)
            # query_embedding = self.embedding_manager.model.encode([request.query])[0].tolist()
            query_embedding = self.embedding_manager.generate_query_embedding([request.query])
            search_filter = self._build_filter(request.file_type, request.file_name)
            search_results = self.vector_storage.client.search(
                collection_name = self.vector_storage.collection_name,
                query_vector = query_embedding,
                limit = request.top_k,
                score_threshold = request.score_threshold,
                query_filter = search_filter
            )
            results = []
            for result in search_results:
                search_result = SearchResult(
                    chunk_id = result.payload.get("original_chunk_id", result.payload.get("chunk_id", "")),
                    file_name = result.payload.get("file_name", ""),
                    file_path = result.payload.get("file_path", ""),
                    file_type = result.payload.get("file_type", ""),
                    chunk_text = result.payload.get("chunk_text", ""),
                    chunk_index = result.payload.get("chunk_index", 0),
                    total_chunks = result.payload.get("total_chunks", 0),
                    score=result.score,
                    metadata=result.payload.get("metadata", {})
                )
                results.append(search_result)
            end = time.monotonic()
            processing_time_ms = (end - start) * 1000
            return QueryResponse(
                query = request.query,
                results = results,
                total_results = len(results),
                processing_time_ms = round(processing_time_ms, 2)
            )

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    def _build_filter(self, file_types: Optional[Union[str, List[str]]] = None, file_names: Optional[Union[str, List[str]]] = None) -> Optional[Filter]:
        conditions = []
        if file_types:
            if isinstance(file_types, str):
                file_types = [file_types]  
            normalized_types = []
            for ft in file_types:
                if not ft.startswith('.'):
                    ft = f'.{ft}'
                normalized_types.append(ft)
            conditions.append(
                FieldCondition(
                    key="file_type",
                    match=MatchValue(any=normalized_types) if len(normalized_types) > 1 else MatchValue(value=normalized_types[0])
                )
            )
        if file_names:
            if isinstance(file_names, str):
                file_names = [file_names]
        
            for name in file_names:
                conditions.append(
                    FieldCondition(
                        key="file_name",
                        match=MatchValue(value=name)
                    )
                )
        return Filter(must=conditions) if conditions else None

query_service = QueryService(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="documents",
)

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Document Query API is running", "status": "healthy"}

@app.get("/health", tags=["Health"])
async def health_check():
    try:
        collections = query_service.vector_storage.client.get_collections()
        collection_exists = any(col.name == query_service.vector_storage.collection_name for col in collections.collections)
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collection_exists": collection_exists,
            "collection_name": query_service.vector_storage.collection_name,
            "embedding_model": query_service.embedding_manager.embedding_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/search", response_model=QueryResponse, tags=["Search"])
async def search_documents(request: QueryRequest):
    return query_service.search(request)

@app.get("/search", response_model=QueryResponse, tags=["Search"])
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results", ge=1, le=100),
    score_threshold: float = Query(0.0, description="Minimum score", ge=0.0, le=1.0)
):
    request = QueryRequest(
        query=q,
        top_k=top_k,
        score_threshold=score_threshold
    )
    return query_service.search(request)

@app.get("/collections/info", tags=["Admin"])
async def get_collection_info():
    try:
        collection_info = query_service.vector_storage.client.get_collection(
            query_service.vector_storage.collection_name
        )
        return {
            "collection_name": query_service.vector_storage.collection_name,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "status": collection_info.status,
            "config": {
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.value
            }
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )