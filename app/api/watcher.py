from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

from app.models.requests import WatchFolderRequest
from app.models.responses import WatchFolderResponse, WatcherStatusResponse
from app.models.entities import WatchedFolderInfo
from app.dependencies import get_query_service
from app.core.query_service import QueryService
from app.core.exceptions import ServiceError, ValidationError


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/watcher", tags=["Watcher"])

@router.post("/start")
async def start_watcher(query_service: QueryService = Depends(get_query_service)):
    """Start the folder watcher service"""
    try:
        result = query_service.start_folder_watching()
        return result
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start watcher: {str(e)}")

@router.post("/stop")
async def stop_watcher(query_service: QueryService = Depends(get_query_service)):
    try:
        result = query_service.stop_folder_watching()
        return result
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop watcher: {str(e)}")

@router.get("/status", response_model=WatcherStatusResponse)
async def get_watcher_status(query_service: QueryService = Depends(get_query_service)):
    try:
        status = query_service.get_folder_watcher_status()
        return WatcherStatusResponse(**status)
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get watcher status: {str(e)}")

@router.post("/folders", response_model=WatchFolderResponse)
async def add_watch_folder(
    request: WatchFolderRequest,
    query_service: QueryService = Depends(get_query_service)
):
    try:
        result = query_service.add_watch_folder(
            folder_path=request.folder_path,
            allowed_extensions=request.allowed_extensions,
            recursive=request.recursive,
            auto_ingest=request.auto_ingest
        )
        return WatchFolderResponse(**result)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add watch folder: {str(e)}")

@router.delete("/folders/{folder_path:path}", response_model=WatchFolderResponse)
async def remove_watch_folder(
    folder_path: str,
    query_service: QueryService = Depends(get_query_service)
):
    try:
        result = query_service.remove_watch_folder(folder_path)
        return WatchFolderResponse(**result)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove watch folder: {str(e)}")

@router.get("/folders", response_model=List[WatchedFolderInfo])
async def get_watched_folders(query_service: QueryService = Depends(get_query_service)):
    try:
        folders = query_service.get_watched_folders()
        return [WatchedFolderInfo(**folder) for folder in folders]
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get watched folders: {str(e)}")

@router.get("/folders/{folder_path:path}", response_model=WatchedFolderInfo)
async def get_watch_folder_info(
    folder_path: str,
    query_service: QueryService = Depends(get_query_service)
):
    try:
        folders = query_service.get_watched_folders()
        for folder in folders:
            if folder['path'] == folder_path:
                return WatchedFolderInfo(**folder)
        
        raise HTTPException(status_code=404, detail=f"Watched folder not found: {folder_path}")
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get folder info: {str(e)}")

# Batch Operations
@router.post("/folders/batch")
async def add_multiple_watch_folders(
    folders: List[WatchFolderRequest],
    query_service: QueryService = Depends(get_query_service)
):
    results = []
    errors = []
    
    for folder_request in folders:
        try:
            result = query_service.add_watch_folder(
                folder_path=folder_request.folder_path,
                allowed_extensions=folder_request.allowed_extensions,
                recursive=folder_request.recursive,
                auto_ingest=folder_request.auto_ingest
            )
            results.append(result)
        except Exception as e:
            errors.append({
                "folder_path": folder_request.folder_path,
                "error": str(e)
            })
    
    return {
        "message": f"Processed {len(folders)} folders",
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None
    }

@router.delete("/folders/batch")
async def remove_multiple_watch_folders(
    folder_paths: List[str],
    query_service: QueryService = Depends(get_query_service)
):
    results = []
    errors = []
    
    for folder_path in folder_paths:
        try:
            result = query_service.remove_watch_folder(folder_path)
            results.append(result)
        except Exception as e:
            errors.append({
                "folder_path": folder_path,
                "error": str(e)
            })
    
    return {
        "message": f"Processed {len(folder_paths)} folders",
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None
    }

@router.post("/folders/{folder_path:path}/force-scan")
async def force_scan_folder(
    folder_path: str,
    query_service: QueryService = Depends(get_query_service)
):
    try:
        from app.models.requests import IngestRequest
        
        request = IngestRequest(
            directory_path=folder_path,
            skip_existing=False,  # Force re-processing
            max_workers=3
        )
        
        result = query_service.ingest(request)
        
        return {
            "message": f"Force scan completed for: {folder_path}",
            "folder_path": folder_path,
            "files_processed": result.files_processed,
            "files_skipped": result.files_skipped,
            "failed_files": result.failed_files,
            "total_chunks": result.total_chunks,
            "processing_time_ms": result.processing_time_ms,
            "status": "success"
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to force scan folder: {str(e)}")

@router.get("/events/queue-size")
async def get_queue_size(query_service: QueryService = Depends(get_query_service)):
    try:
        status = query_service.get_folder_watcher_status()
        return {
            "queue_size": status["queue_size"],
            "is_running": status["is_running"],
            "timestamp": status.get("timestamp", None)
        }
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue size: {str(e)}")