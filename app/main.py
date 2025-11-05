"""
Main FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import settings
from app.api import health, search, ingest, collections

def create_app() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(ingest.router)
    app.include_router(collections.router)
    
    return app

app = create_app()

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/api")
async def api_info():
    return {
        "message": f"{settings.app_name} is running",
        "status": "healthy",
        "version": settings.app_version,
        "docs_url": "/docs",
        "web_interface": "/"
    }