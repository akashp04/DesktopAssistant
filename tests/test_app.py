import warnings
warnings.filterwarnings("ignore")

def test_app():
    print("Testing Desktop Assistant Application")
    
    try:
        print("Loading configuration...")
        from app.config import settings
        print(f"Config loaded: Qdrant at {settings.qdrant_path}")
        
        print("Creating FastAPI app...")
        from app.main import create_app
        app = create_app()
        print("FastAPI app created successfully")
        
        print("Testing query service...")
        from app.core.query_service import QueryService
        service = QueryService()
        print("Query service initialized")
        
        print("Testing document search (empty query)...")
        from app.models.requests import QueryRequest
        query = QueryRequest(query="Harry Potter",top_k=5, score_threshold=0.0)
        results = service.search(query)
        print(f"Search returned {results.total_results} results")
        print(f"Here are the Results:\n{results.results}")
        
        print("\nAll tests passed!")
        print("Application is working correctly!")
        print(f"Qdrant storage: {settings.qdrant_path}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_app()