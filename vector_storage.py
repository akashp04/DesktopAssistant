from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List
from dataclasses import asdict
import uuid
import os 

from document_chunker import DocumentChunk

class VectorStorage:
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6333,
                 collection_name: str = "documents",
                 vector_size: int = 384,
                 path: str = None
                ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.vector_size = vector_size
        self.path = path
        self._load_client()
        self._initialize_collection()

    def _load_client(self):
        try:
            if self.path:
                os.makedirs(self.path, exist_ok=True)
                self.client = QdrantClient(path=self.path)
                print(f"Connected to Qdrant at path: {self.path}")
            else:
                self.client = QdrantClient(host=self.host, port=self.port)
                print(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise e

    def _initialize_collection(self):
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name = self.collection_name,
                    vectors_config = VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                print(f"No Collection {self.collection_name} found\nCreated new collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists.")
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            raise e
    
    def store_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")
        
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            uuid_chunk_id = self._string_uuid(chunk.chunk_id)
            point = PointStruct(
                id=uuid_chunk_id,
                vector=embedding,
                payload={
                    **asdict(chunk), 
                    "chunk_text_preview": chunk.chunk_text[:200] + "..." if len(chunk.chunk_text) > 200 else chunk.chunk_text,
                    "original_chunk_id": chunk.chunk_id
                }
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Stored {len(points)} chunks in collection {self.collection_name}.")
        except Exception as e:
            print(f"Error storing chunks: {e}")
            raise e
        
    def _string_uuid(self, string: str) -> str:
        namespace = uuid.UUID('12345678-1234-5678-1234-567812345678')
        return str(uuid.uuid5(namespace, string))
    
    def file_exists(self, file_hash: str) -> bool:
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {"key": "file_hash", "match": {"value": file_hash}}
                    ]
                },
                limit=1
            )
            return len(result[0]) > 0
        except Exception as e:
            print(f"Error checking file existence: {e}")
            raise e
    
    def delete_collection(self) -> bool:
        try:
            if not self.client.collection_exists(self.collection_name):
                print(f"Collection {self.collection_name} does not exist.")
                return False
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            raise e

    def get_collection_info(self):
        try:
            if not self.client.collection_exists(self.collection_name):
                return {"exists": False, "message": f"Collection {self.collection_name} does not exist"}
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return {"exists": True, 
                    "name":self.collection_name,
                    "points_count": collection_info.points_count,
                    "vectors_count": collection_info.vectors_count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance_metric": collection_info.config.params.vectors.distance.value
                }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            raise e