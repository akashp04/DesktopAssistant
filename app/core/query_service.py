from DesktopAssistant.storage.embedding_manager import EmbeddingManager
from DesktopAssistant.storage.vector_storage import VectorStorage
from DesktopAssistant.storage.pipeline import Pipeline
from DesktopAssistant.app.models.requests import QueryRequest, IngestRequest
from DesktopAssistant.app.models.responses import QueryResponse, IngestResponse
from DesktopAssistant.app.models.entities import SearchResult

import time
import os
from typing import List, Optional, Union
import logging
from fastapi import HTTPException
from qdrant_client.models import Filter, FieldCondition, MatchValue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            search_results = self.vector_storage.search_query(
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

    def ingest(self, request: IngestRequest) -> IngestResponse:
        if not os.path.exists(request.directory_path):
            raise HTTPException(status_code=400, detail=f"Directory does not exist: {request.directory_path}")
        if not os.path.isdir(request.directory_path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.directory_path}")
        if request.chunk_size <= request.overlap:
            raise HTTPException(status_code=400, detail="Chunk size must be greater than overlap.")
        
        start = time.monotonic()
        try:
            self.pipeline = Pipeline(
                base_directory = request.directory_path,
                embedding_model = self.embedding_manager.embedding_model,
                vector_storage = self.vector_storage,
                chunk_size = request.chunk_size,
                overlap = request.overlap,
            )
            result = self.pipeline.index_directory(skip_existing = request.skip_existing)
            end = time.monotonic()
            processing_time_ms = (end - start) * 1000
            return IngestResponse(
                message = "Ingestion completed successfully",
                directory_path = request.directory_path,
                files_processed = result['processed_files'],
                files_skipped = result['skipped_files'],
                failed_files = result['failed_files'],
                total_chunks = result['total_chunks'],
                processing_time_ms = round(processing_time_ms, 2),
                status = "success"
            )
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
