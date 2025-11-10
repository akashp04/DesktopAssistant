"""
Search endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from app.dependencies import get_query_service
from app.models.requests import QueryRequest, HybridSearchRequest
from app.models.responses import QueryResponse, HybridSearchResponse
from app.core.query_service import QueryService
from app.core.exceptions import ServiceError, ValidationError

'''
TODO:
Add Hybrid Search Endpoint
'''

router = APIRouter(prefix="/search", tags=["Search"])

@router.post("/", response_model=QueryResponse)
async def search_documents(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service)
):
    try:
        return query_service.search(request)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/semantic", response_model=QueryResponse)
async def semantic_search(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service)
):
    """Pure semantic search using vector embeddings"""
    try:
        return await query_service.search(request)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hybrid", response_model=HybridSearchResponse)
def hybrid_search(
    request: HybridSearchRequest,
    query_service: QueryService = Depends(get_query_service)
):
    try:
        return query_service.hybrid_search(request)
    except ValidationError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rebuild-index")
def rebuild_keyword_index(query_service: QueryService = Depends(get_query_service)):
    try:
        return query_service.rebuild_keyword_index()
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))