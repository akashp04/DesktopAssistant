"""
Dependency injection for FastAPI
"""
from functools import lru_cache
from app.core.query_service import QueryService

@lru_cache()
def get_query_service() -> QueryService:
    return QueryService()