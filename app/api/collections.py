from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_query_service
from app.models.responses import CreateCollectionResponse, DeleteCollectionResponse
from app.core.query_service import QueryService
from app.core.exceptions import ServiceError
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/collections", tags=["Admin"])

'''
TO-DO:
Change the get_collection_details functions as creating a new temp collection when requesting info
'''


@router.get("/info", tags=["Admin"])
async def get_collection_info(query_service: QueryService = Depends(get_query_service)):
    try:
        info = query_service.get_collection_info()
        return info
    except ServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{collection_name}/info", tags=["Admin"])
async def get_collection_details(
    collection_name: str,
    query_service: QueryService = Depends(get_query_service)
):
    try:
        info = query_service.get_collection_info(collection_name) 
        return info
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{collection_name}/clear", tags=["Admin"])
async def clear_collection(
    collection_name: str,
    query_service: QueryService = Depends(get_query_service)
):
    try:
        cleared = query_service.clear_collection()
        
        return {
            "message": f"Collection '{collection_name}' cleared successfully" if cleared else f"Collection '{collection_name}' was already empty",
            "collection_name": collection_name,
            "cleared": cleared,
            "status": "success"
        }
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{collection_name}", response_model=DeleteCollectionResponse, tags=["Admin"])
async def delete_collection(
    collection_name: str,
    query_service: QueryService = Depends(get_query_service)
):
    try:
        deleted = query_service.delete_collection()
        
        return DeleteCollectionResponse(
            message=f"Collection '{collection_name}' {'deleted' if deleted else 'cleared'} successfully.",
            collection_name=collection_name,
            deleted=deleted,
            status="success"
        )
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))