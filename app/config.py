from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "Desktop Assistant"
    app_version: str = "1.0.0"
    app_description: str = "API for managing and querying desktop assistant embeddings and vectors"

    host: str = "localhost"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_path: Optional[str] = "./qdrant_storage"  # Changed to use path mode
    collection_name: str = "documents"

    embedding_model: str = "BAAI/bge-small-en-v1.5"  # Fixed field name
    vector_size: int = 384

    default_chunk_size: int = 450  # Fixed field name
    default_overlap: int = 100     # Fixed field name
    max_workers: int = 3

    top_k: int = 5
    max_top_k: int = 100
    score_threshold: float = 0.0

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()