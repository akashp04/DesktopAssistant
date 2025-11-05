from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import List, Optional, Dict, Any, Union
import time
import os


from DesktopAssistant.storage.embedding_manager import EmbeddingManager
from DesktopAssistant.storage.vector_storage import VectorStorage
from DesktopAssistant.storage.pipeline import Pipeline
from DesktopAssistant.app.models.requests import QueryRequest, IngestRequest
from DesktopAssistant.app.models.responses import QueryResponse, IngestResponse, CreateCollectionResponse, DeleteCollectionResponse
from DesktopAssistant.app.models.entities import SearchResult
from DesktopAssistant.app.core.query_service import QueryService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Desktop Assistant API",
    description="API for managing and querying desktop assistant embeddings and vectors.",
    version="1.0.0"
)

# class QueryRequest(BaseModel):
#     query: str = Field(..., description="The query string to search for.", min_length=1)
#     top_k: int = Field(5, description="Number of top results to return.", ge=1, le=100)
#     score_threshold: float = Field(0.0, description="Minimum score threshold for results.", ge=0.0, le=1.0)
#     file_type: Optional[Union[ str, List[str]]] = Field(None, description="Filter results by file extensions (e.g., 'pdf', 'txt').")
#     file_name: Optional[Union[str, List[str]]] = Field(None, description="Filter results by file name substring")

# class SearchResult(BaseModel):
#     chunk_id: str
#     file_name: str
#     file_path: str
#     file_type: str
#     chunk_text: str
#     chunk_index: int
#     total_chunks: int
#     score: float
#     metadata: Dict[str, Any]

# class QueryResponse(BaseModel):
#     query: str
#     results: List[SearchResult]
#     total_results: int
#     processing_time_ms: float

# class IngestRequest(BaseModel):
#     directory_path: str = Field(..., description="Path to the directory to ingest files from.", min_length=1)
#     skip_existing: bool = Field(True, description="Whether to skip files that are already indexed.")
#     chunk_size: int = Field(400, description="Chunk size for document chunking.", ge=1, le=512)
#     overlap: int = Field(100, description="Overlap size for document chunking.", ge=0, le=511)

# class IngestResponse(BaseModel):
#     message: str
#     directory_path: str
#     files_processed: int
#     files_skipped: int
#     failed_files: int
#     total_chunks: int
#     processing_time_ms: float
#     status: str

# class CreateCollectionResponse(BaseModel):
#     message: str
#     collection_name: str
#     created: bool
#     status: str

# class DeleteCollectionResponse(BaseModel):
#     message: str
#     collection_name: str
#     deleted: bool
#     status: str

# class QueryService:
#     def __init__(self, 
#                  qdrant_host: str = "localhost",
#                  qdrant_port: int = 6333,
#                  qdrant_path: str = None,
#                  collection_name: str = "documents",
#                  embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        
#         self.embedding_manager = EmbeddingManager(embedding_model = embedding_model_name)
#         self.vector_storage = VectorStorage(
#             host=qdrant_host,
#             port=qdrant_port,
#             path=qdrant_path,
#             collection_name=collection_name
#         )
#         logger.info ("Query Service initialized.")
    
#     def search(self, request: QueryRequest) -> QueryResponse:
#         start = time.monotonic()
#         try:
#             # query_embedding = self.embedding_manager.generate_embedding(request.query)
#             # query_embedding = self.embedding_manager.model.encode([request.query])[0].tolist()
#             query_embedding = self.embedding_manager.generate_query_embedding([request.query])
#             search_filter = self._build_filter(request.file_type, request.file_name)
#             search_results = self.vector_storage.search_query(
#                 collection_name = self.vector_storage.collection_name,
#                 query_vector = query_embedding,
#                 limit = request.top_k,
#                 score_threshold = request.score_threshold,
#                 query_filter = search_filter
#             )
#             results = []
#             for result in search_results:
#                 search_result = SearchResult(
#                     chunk_id = result.payload.get("original_chunk_id", result.payload.get("chunk_id", "")),
#                     file_name = result.payload.get("file_name", ""),
#                     file_path = result.payload.get("file_path", ""),
#                     file_type = result.payload.get("file_type", ""),
#                     chunk_text = result.payload.get("chunk_text", ""),
#                     chunk_index = result.payload.get("chunk_index", 0),
#                     total_chunks = result.payload.get("total_chunks", 0),
#                     score=result.score,
#                     metadata=result.payload.get("metadata", {})
#                 )
#                 results.append(search_result)
#             end = time.monotonic()
#             processing_time_ms = (end - start) * 1000
#             return QueryResponse(
#                 query = request.query,
#                 results = results,
#                 total_results = len(results),
#                 processing_time_ms = round(processing_time_ms, 2)
#             )

#         except Exception as e:
#             logger.error(f"Error during search: {e}")
#             raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
#     def _build_filter(self, file_types: Optional[Union[str, List[str]]] = None, file_names: Optional[Union[str, List[str]]] = None) -> Optional[Filter]:
#         conditions = []
#         if file_types:
#             if isinstance(file_types, str):
#                 file_types = [file_types]  
#             normalized_types = []
#             for ft in file_types:
#                 if not ft.startswith('.'):
#                     ft = f'.{ft}'
#                 normalized_types.append(ft)
#             conditions.append(
#                 FieldCondition(
#                     key="file_type",
#                     match=MatchValue(any=normalized_types) if len(normalized_types) > 1 else MatchValue(value=normalized_types[0])
#                 )
#             )
#         if file_names:
#             if isinstance(file_names, str):
#                 file_names = [file_names]
        
#             for name in file_names:
#                 conditions.append(
#                     FieldCondition(
#                         key="file_name",
#                         match=MatchValue(value=name)
#                     )
#                 )
#         return Filter(must=conditions) if conditions else None

#     def ingest(self, request: IngestRequest) -> IngestResponse:
#         if not os.path.exists(request.directory_path):
#             raise HTTPException(status_code=400, detail=f"Directory does not exist: {request.directory_path}")
#         if not os.path.isdir(request.directory_path):
#             raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.directory_path}")
#         if request.chunk_size <= request.overlap:
#             raise HTTPException(status_code=400, detail="Chunk size must be greater than overlap.")
        
#         start = time.monotonic()
#         try:
#             self.pipeline = Pipeline(
#                 base_directory = request.directory_path,
#                 embedding_model = self.embedding_manager.embedding_model,
#                 vector_storage = self.vector_storage,
#                 chunk_size = request.chunk_size,
#                 overlap = request.overlap,
#             )
#             result = self.pipeline.index_directory(skip_existing = request.skip_existing)
#             end = time.monotonic()
#             processing_time_ms = (end - start) * 1000
#             return IngestResponse(
#                 message = "Ingestion completed successfully",
#                 directory_path = request.directory_path,
#                 files_processed = result['processed_files'],
#                 files_skipped = result['skipped_files'],
#                 failed_files = result['failed_files'],
#                 total_chunks = result['total_chunks'],
#                 processing_time_ms = round(processing_time_ms, 2),
#                 status = "success"
#             )
#         except Exception as e:
#             logger.error(f"Error during ingestion: {e}")
#             raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

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

# @app.get("/search", response_model=QueryResponse, tags=["Search"])
# async def search_documents_get(
#     q: str = Query(..., description="Search query"),
#     top_k: int = Query(5, description="Number of results", ge=1, le=100),
#     score_threshold: float = Query(0.0, description="Minimum score", ge=0.0, le=1.0)
# ):
#     request = QueryRequest(
#         query=q,
#         top_k=top_k,
#         score_threshold=score_threshold
#     )
#     return query_service.search(request)
@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest_documents(request: IngestRequest):
    return query_service.ingest(request)

@app.get("/collections/info", tags=["Admin"])
async def get_collection_info():
    try:
        collection_info = query_service.vector_storage.get_collection_info()
        return collection_info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

@app.post("/collections/{collection_name}/create", response_model=CreateCollectionResponse, tags=["Admin"])
async def create_collection(collection_name: str):
    try:
        temp_storage = VectorStorage(
            host=query_service.vector_storage.host,
            port=query_service.vector_storage.port,
            path=query_service.vector_storage.path,
            collection_name=collection_name,
            vector_size=384  
        )
        
        return CreateCollectionResponse(
            message=f"Collection '{collection_name}' created successfully",
            collection_name=collection_name,
            created=True,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")
    
@app.get("/collections/{collection_name}/info", tags=["Admin"])
async def get_collection_details(collection_name: str):
    try:
        temp_storage = VectorStorage(
            host=query_service.vector_storage.host,
            port=query_service.vector_storage.port,
            path=query_service.vector_storage.path,
            collection_name=collection_name
        )

        collection_info = temp_storage.get_collection_info()
        return collection_info
    except Exception as e:
        logger.error(f"Error getting collection info for {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")


@app.delete("/collections/{collection_name}", response_model=DeleteCollectionResponse, tags=["Admin"])
async def delete_collection(collection_name: str):
    try:
        temp_storage = VectorStorage(
            host=query_service.vector_storage.host,
            port=query_service.vector_storage.port,
            path=query_service.vector_storage.path,
            collection_name=collection_name
        )
        deleted = temp_storage.delete_collection()
        return DeleteCollectionResponse(
            message = f"Collection '{collection_name}' deleted successfully." if deleted else f"Collection '{collection_name}' does not exist.",
            collection_name = collection_name,
            deleted = deleted,
            status = "success" if deleted else "not_found"
        )
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )