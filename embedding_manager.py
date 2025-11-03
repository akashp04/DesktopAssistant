from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import threading
from threading import Lock

from document_chunker import DocumentChunk

class EmbeddingManager:
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        self.model = None
        self.embedding_model = embedding_model
        self.model_lock = Lock()
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.embedding_model)
            dimension = self.model.get_sentence_embedding_dimension()
            print(f"Loaded embedding model: {self.embedding_model} with Embedding Size: {dimension}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise e

    def generate_embedding(self, chunks: List[DocumentChunk]) -> np.ndarray:
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        if not chunks:
            raise ValueError("No document chunks provided for embedding generation.")
        try:
            text = [chunk.chunk_text for chunk in chunks]
            with self.model_lock:
                embeddings = self.model.encode(
                    text, 
                    batch_size=8,
                    show_progress_bar=False, 
                    convert_to_tensor=False,
                    normalize_embeddings=True
                )
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise e
        
    def generate_query_embedding(self, query: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        try:
            embedding = self.model.encode(query)
            return embedding[0].tolist()
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            raise e