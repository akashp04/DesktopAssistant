"""
Document ingestion endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_query_service
from app.models.requests import IngestRequest
from app.models.responses import IngestResponse
from app.core.query_service import QueryService
from app.core.exceptions import ServiceError, ValidationError

router = APIRouter(prefix="/ingest", tags=["Admin"])

@router.post("/", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    query_service: QueryService = Depends(get_query_service)
):
    try:
        return query_service.ingest(request)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))