from embedding_manager import EmbeddingManager
from vector_storage import VectorStorage
from document_chunker import DocumentChunker
from file_handler import FileHandler

from typing import List
import argparse




class Pipeline:
    def __init__(self, 
                 base_directory: str,
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 qdrant_path: str = None,
                 chunk_size: int = 400,
                 overlap: int = 100,
                 ):
        self.loader = FileHandler(base_directory)
        self.chunker = DocumentChunker(
            chunk_size = chunk_size, 
            overlap = overlap,
            file_handler = self.loader,
        )
        self.embedding = EmbeddingManager(embedding_model=embedding_model)
        self.vector_storage = VectorStorage(path = qdrant_path)
    
    def index_directory(self, skip_existing: bool = True):
        files = self.loader.get_all_files()
        total_chunks = 0 
        processed_files = 0
        skipped_files = 0

        for i, file_path in enumerate(files):
            print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
            file_hash = self.loader.compute_hash(str(file_path))
            
            if skip_existing and self.vector_storage.file_exists(file_hash):
                print(f"Skipping {file_path.name}, already indexed.")
                skipped_files += 1
                continue
            
            elements = self.loader.extract_text_from_file(str(file_path))
            if not elements:
                continue
            total_text_length = sum(len(str(el)) for el in elements)
            print(f"Extracted {len(elements)} elements ({total_text_length} characters)")

            chunks = self.chunker.chunk_text(elements, file_path)
            if not chunks:
                print(f"No chunks created for {file_path.name}, skipping.")
                continue

            print(f"Created {len(chunks)} chunks")
            # if chunks[0].heading:
            #     print(f" 
            #   First heading: '{chunks[0].heading[:50]}...'")
            # if chunks[0].page_number:
            #     print(f"   Page range: {chunks[0].page_number} - {chunks[-1].page_number}")
            # if chunks[0].element_types:
            #     unique_types = set()
            #     for chunk in chunks:
            #         if chunk.element_types:
            #             unique_types.update(chunk.element_types)
            #     print(f"   Element types: {', '.join(sorted(unique_types))}")
            
            embeddings = self.embedding.generate_embedding(chunks)
            self.vector_storage.store_chunks(chunks, embeddings)
            total_chunks += len(chunks)
            processed_files += 1
        
        print(f"\n{'='*60}")
        print(f"  Indexing complete!")
        print(f"   Files processed: {processed_files}")
        print(f"   Files skipped: {skipped_files}")
        print(f"   Total chunks: {total_chunks}")
        print(f"{'='*60}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Indexing Pipeline")
    parser.add_argument("--base_directory", type=str, required=True, help="Base directory containing documents to index")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model to use")
    parser.add_argument("--qdrant_path", type=str, default=None, help="Path to Qdrant database")
    parser.add_argument("--chunk_size", type=int, default=500, help="Number of tokens per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Number of overlapping tokens between chunks")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files that are already indexed")
    
    args = parser.parse_args()
    
    pipeline = Pipeline(
        base_directory=args.base_directory,
        embedding_model=args.embedding_model,
        qdrant_path=args.qdrant_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    
    pipeline.index_directory(skip_existing=args.skip_existing)