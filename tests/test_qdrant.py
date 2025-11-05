import os
import warnings

warnings.filterwarnings("ignore", message=".*GetPrototype.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_qdrant_path():
    try:
        print("🔧 Testing Qdrant path configuration...")
        
        from app.config import settings
        print(f"✅ Config loaded: path={settings.qdrant_path}")
        
        if settings.qdrant_path:
            storage_path = settings.qdrant_path
            if not os.path.exists(storage_path):
                os.makedirs(storage_path, exist_ok=True)
                print(f"Created storage directory: {storage_path}")
            else:
                print(f"Storage directory exists: {storage_path}")
            
            from qdrant_client import QdrantClient
            client = QdrantClient(path=storage_path)
            collections = client.get_collections()
            print(f"Qdrant client connected successfully")
            print(f"Found {len(collections.collections)} existing collections")
            
            for col in collections.collections:
                print(f"   - {col.name}")
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_full_app():
    try:
        print("\nTesting full application import...")
        from app.main import app
        print("Application imported successfully")
        
        from app.dependencies import get_query_service
        service = get_query_service()
        print("Query service created successfully")
        
        return True
        
    except Exception as e:
        print(f"Full app error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Desktop Assistant - Qdrant Path Test\n")
    
    path_ok = test_qdrant_path()
    
    if path_ok:
        app_ok = test_full_app()
        
        if app_ok:
            print("\nAll tests passed! Starting application...")
            import uvicorn
            from app.config import settings
            
            uvicorn.run(
                "app.main:app",
                host=settings.host,
                port=settings.port,
                reload=settings.reload,
                log_level=settings.log_level
            )
        else:
            print("\nApp import failed, but Qdrant path is configured correctly.")
    else:
        print("\nQdrant path configuration failed.")