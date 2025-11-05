"""
Application startup script
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.app_name} on http://{settings.host}:{settings.port}")
    print(f"Using Qdrant at: {settings.qdrant_path}")
    
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level
    )