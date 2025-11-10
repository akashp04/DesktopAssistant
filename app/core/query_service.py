import time
import os
from typing import List, Optional, Union, Dict, Any

from app.config import settings
from app.models.requests import QueryRequest, IngestRequest
from app.models.responses import QueryResponse, IngestResponse, HybridSearchResponse
from app.models.entities import SearchResult
from app.core.exceptions import ServiceError, ValidationError
from storage.embedding_manager import EmbeddingManager
from storage.vector_storage import VectorStorage
from storage.pipeline import Pipeline
from storage.hybrid_searcher import HybridSearch, BM25Search, ExactMatchSearch, RRF

import logging
from fastapi import HTTPException, UploadFile
from qdrant_client.models import Filter, FieldCondition, MatchValue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self):
        self.embedding_manager = EmbeddingManager(embedding_model=settings.embedding_model)
        self.vector_storage = VectorStorage(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            path=settings.qdrant_path,
            collection_name=settings.collection_name,
            vector_size=settings.vector_size
        )
        logger.info("Query Service initialized.")

        self.bm25_search = BM25Search()
        self.exact_match_search = ExactMatchSearch()
        self.rrf_ranker = RRF()
        
        self.hybrid_searcher = HybridSearch(
            vector_storage=self.vector_storage,
            embedding_manager=self.embedding_manager,
            bm25=self.bm25_search,
            exact_matcher=self.exact_match_search,
            rrf_ranker=self.rrf_ranker
        )
        
        self._folder_watcher = None
    
    @property
    def folder_watcher(self):
        if self._folder_watcher is None:
            from watchservice.watcherservice import FolderWatcherService
            self._folder_watcher = FolderWatcherService(
                query_service=self,
                batch_size=getattr(settings, 'WATCHER_BATCH_SIZE', 5),
                poll_interval=getattr(settings, 'WATCHER_POLL_INTERVAL', 30000.0)
            )
        return self._folder_watcher
    
    def search(self, request: QueryRequest) -> QueryResponse:
        start = time.monotonic()
        try:
            # query_embedding = self.embedding_manager.generate_embedding(request.query)
            # query_embedding = self.embedding_manager.model.encode([request.query])[0].tolist()
            query_embedding = self.embedding_manager.generate_query_embedding([request.query])
            search_filter = self._build_filter(request.file_type, request.file_name)
            search_results = self.vector_storage.search_query(
                collection_name = self.vector_storage.collection_name,
                query_embedding = query_embedding,
                top_k = request.top_k,
                score_threshold = request.score_threshold,
                search_filter = search_filter
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
                    semantic_score=result.score,
                    final_score=result.score,
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
    
    def hybrid_search(self, request: QueryRequest) -> QueryResponse:
        start = time.monotonic()
        try:
            if not self.hybrid_searcher.corpus_built:
                self.hybrid_searcher.build_keyword_index()
            
            search_filter = self._build_filter(request.file_type, request.file_name)
            results = self.hybrid_searcher.hybrid_search(
                query = request.query,
                top_k = request.top_k,
                semantic_weight = request.semantic_weight,
                keyword_weight = request.keyword_weight,
                exact_weight = request.exact_weight,
                search_filter = search_filter
            )
            end = time.monotonic()
            processing_time_ms = (end - start) * 1000

            search_breakdown = {
                'semantic_matches': len([r for r in results if r.semantic_score > 0]),
                'keyword_matches': len([r for r in results if r.keyword_score > 0]),
                'exact_matches': len([r for r in results if r.exact_match_score > 0])
            }

            return HybridSearchResponse(
                query=request.query,
                results=results,
                total_results=len(results),
                processing_time_ms=round(processing_time_ms, 2),
                search_breakdown=search_breakdown
            )
        except Exception as e:
            raise ServiceError(f"Hybrid search failed: {str(e)}")
    
    def rebuild_keyword_index(self) -> Dict[str, Any]:
        try:
            start = time.monotonic()
            self.hybrid_searcher.build_keyword_index(force_rebuild=True)
            end = time.monotonic()
            
            return {
                "message": "Keyword index rebuilt successfully",
                "documents_indexed": len(self.hybrid_searcher.document_corpus),
                "rebuild_time_ms": round((end - start) * 1000, 2),
                "status": "success"
            }
        except Exception as e:
            raise ServiceError(f"Failed to rebuild keyword index: {str(e)}")
    

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
        
    def create_collection(self, collection_name: str = 'documents') -> bool:
        try:
            result = self.vector_storage.create_collection(collection_name)
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                return False
            else:
                raise ServiceError(f"Failed to create collection {collection_name}: {str(e)}")
            
    def get_collection_info(self, collection_name: str = 'documents'):
        try:
            info = self.vector_storage.get_collection_info()
            return info
        except Exception as e:
            logger.error(f"Error during getting Collection Info: {e}")
            raise HTTPException(status_code=500, detail=f"Retrieving Collection Info Failed: {str(e)}")

    def clear_collection(self) -> bool:
        try:
            info = self.vector_storage.get_collection_info()
            if not info.get("exists", False):
                return False
            
            points_count = info.get("points_count", 0)
            if points_count == 0:
                return False 
            
            self.vector_storage.delete_collection()
            self.vector_storage._initialize_collection()
            
            logger.info(f"Cleared collection: {self.vector_storage.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise HTTPException(status_code=500, detail=f"Clearing collection failed: {str(e)}")

    def delete_collection(self) -> bool:
        try:
            deleted = self.vector_storage.delete_collection()
            if deleted:
                logger.info(f"Deleted collection: {self.vector_storage.collection_name}")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise HTTPException(status_code=500, detail=f"Deleting collection failed: {str(e)}")
    
    def start_folder_watching(self) -> Dict[str, Any]:
        try:
            self.folder_watcher.start_watching()
            return {"message": "Folder watching started", "status": "success"}
        except Exception as e:
            raise ServiceError(f"Failed to start folder watching: {str(e)}")

    def stop_folder_watching(self) -> Dict[str, Any]:
        try:
            self.folder_watcher.stop_watching()
            return {"message": "Folder watching stopped", "status": "success"}
        except Exception as e:
            raise ServiceError(f"Failed to stop folder watching: {str(e)}")

    def add_watch_folder(self, folder_path: str, allowed_extensions: List[str] = None, 
                        recursive: bool = True, auto_ingest: bool = True) -> Dict[str, Any]:
        try:
            success = self.folder_watcher.add_watch_folder(
                folder_path=folder_path,
                allowed_extensions=allowed_extensions,
                recursive=recursive,
                auto_ingest=auto_ingest
            )
            
            if success:
                return {
                    "message": f"Added watch folder: {folder_path}",
                    "folder_path": folder_path,
                    "status": "success"
                }
            else:
                raise ServiceError("Failed to add watch folder")
                
        except Exception as e:
            raise ServiceError(f"Failed to add watch folder: {str(e)}")

    def remove_watch_folder(self, folder_path: str) -> Dict[str, Any]:
        try:
            success = self.folder_watcher.remove_watch_folder(folder_path)
            
            if success:
                return {
                    "message": f"Removed watch folder: {folder_path}",
                    "folder_path": folder_path,
                    "status": "success"
                }
            else:
                raise ServiceError("Watch folder not found")
                
        except Exception as e:
            raise ServiceError(f"Failed to remove watch folder: {str(e)}")

    def get_folder_watcher_status(self) -> Dict[str, Any]:
        try:
            return self.folder_watcher.get_watch_status()
        except Exception as e:
            raise ServiceError(f"Failed to get folder watcher status: {str(e)}")

    def get_watched_folders(self) -> List[Dict[str, Any]]:
        try:
            return self.folder_watcher.get_watched_folders()
        except Exception as e:
            raise ServiceError(f"Failed to get watched folders: {str(e)}")
    