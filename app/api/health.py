"""
Health check endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
import time
from app.dependencies import get_query_service
from app.models.responses import HealthResponse
from app.core.query_service import QueryService

router = APIRouter(prefix="/health", tags=["Health"])

start_time = time.time()

@router.get("/", response_model=HealthResponse)
async def health_check(query_service: QueryService = Depends(get_query_service)):
    try:
        collections = query_service.vector_storage.client.get_collections()
        collection_exists = any(
            col.name == query_service.vector_storage.collection_name 
            for col in collections.collections
        )
        
        return HealthResponse(
            status="healthy",
            qdrant_connected=True,
            collection_exists=collection_exists,
            collection_name=query_service.vector_storage.collection_name,
            embedding_model=query_service.embedding_manager.embedding_model,
            uptime_seconds=round(time.time() - start_time, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")