from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
from tiktoken import get_encoding, Encoding

from file_handler import FileHandler


@dataclass   
class DocumentChunk:
    chunk_id: str
    file_path: str
    file_name: str
    file_type: str
    chunk_text: str
    chunk_index: int
    total_chunks: int
    file_hash: str
    modified_time: float
    metadata: Dict

class DocumentChunker:
    MAX_CHUNK_SIZE = 512
    def __init__(self, chunk_size: int = 800, overlap: int = 150, file_handler: FileHandler = None, tokenizer: Encoding = None):
        assert chunk_size <= self.MAX_CHUNK_SIZE, f"Chunk size cannot exceed {self.MAX_CHUNK_SIZE} tokens."
        assert overlap < chunk_size, "Overlap must be smaller than chunk size."

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.file_handler = file_handler
        if not tokenizer:
            self.tokenizer = get_encoding("cl100k_base")
            print("Using default tokenizer: cl100k_base")
        else:
            self.tokenizer = tokenizer

    def chunk_text(self, text: str, file_path: str) -> List[DocumentChunk]:
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        chunk_id, start = 0, 0
        
        file_hash = self.file_handler.compute_hash(file_path)
        stat = file_path.stat()

        tokens = self.tokenizer.encode(text)

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunk = {
                'chunk_id': chunk_id,
                'start_token': start,
                'end_token': end,
                'text': chunk_text,
                'token_length': len(chunk_text),
            }
            chunks.append(chunk)
            chunk_id += 1
            start += self.chunk_size - self.overlap
        
        total_chunks = len(chunks)
        doc_chunk_objects = []

        for chunk in chunks:
            doc_chunk = DocumentChunk(
                chunk_id=f"{file_hash}_{chunk['chunk_id']}",
                file_path=str(file_path.absolute()),
                file_name=file_path.name,
                file_type=file_path.suffix,
                chunk_index=chunk['chunk_id'],
                chunk_text=chunk['text'],
                total_chunks=total_chunks,
                file_hash=file_hash,
                modified_time=stat.st_mtime,
                metadata={
                    "size_bytes": stat.st_size,
                    "indexed_at": datetime.now().isoformat(),
                    "start_token": chunk['start_token'],
                    "end_token": chunk['end_token'],
                    "token_count": chunk['token_length']
                }
            )
            doc_chunk_objects.append(doc_chunk)
        return doc_chunk_objects
