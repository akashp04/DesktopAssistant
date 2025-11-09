"""
Document ingestion endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile
from app.dependencies import get_query_service
from app.models.requests import IngestRequest
from app.models.responses import IngestResponse
from app.core.query_service import QueryService
from app.core.exceptions import ServiceError, ValidationError
from app.config import settings
import os
import shutil
from typing import List

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
            
@router.post("/upload", response_model=IngestResponse, tags=["Documents"])
async def upload_document(file: UploadFile, query_service: QueryService = Depends(get_query_service)):
    file_path = None
    try:
        if not file.filename:
            raise ValidationError("No file uploaded")
        
        upload_dir = "./uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        request = IngestRequest(
            directory_path=upload_dir,
            skip_existing=False,
            max_workers=1
        )
        return query_service.ingest(request)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(F"Warning: Failed to delete temporary file {file_path}. Error: {e}")

@router.post("/upload-multiple", response_model=IngestResponse, tags=["Documents"])
async def upload_multiple_document(files: List[UploadFile], query_service: QueryService = Depends(get_query_service)):
    temp_files = []
    try:
        if not files:
            raise ValidationError("No files provided")

        upload_dir = "./uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        for file in files:
            if not file.filename:
                continue
            file_path = os.path.join(upload_dir, file.filename)
            temp_files.append(file_path)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
        if not temp_files:
            raise ValidationError("No files uploaded.")
        
        request = IngestRequest(
            directory_path=upload_dir,
            skip_existing=False,
            max_workers=min(settings.max_workers, len(files))
        )
        return query_service.ingest(request)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(F"Warning: Failed to delete temporary file {file_path}. Error: {e}")
