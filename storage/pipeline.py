from storage.embedding_manager import EmbeddingManager
from storage.vector_storage import VectorStorage
from storage.document_chunker import DocumentChunker
from storage.file_handler import FileHandler

from pathlib import Path
from typing import List, Union
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import argparse
import threading



class Pipeline:
    def __init__(self, 
                 base_directory: str,
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 qdrant_path: str = None,
                 chunk_size: int = 400,
                 overlap: int = 100,
                 vector_storage: VectorStorage = None,
                 max_workers: int = 3
                 ):
        self.max_workers = max_workers
        self.process_lock = Lock()
        self.stats = {
            'processed_files': 0,
            'skipped_files': 0, 
            'failed_files': 0,
            'total_chunks': 0,
            'total_files': 0
        }
        self.loader = FileHandler(base_directory)
        self.chunker = DocumentChunker(
            chunk_size = chunk_size, 
            overlap = overlap,
            file_handler = self.loader,
        )
        self.embedding = EmbeddingManager(embedding_model=embedding_model)
        self.vector_storage = VectorStorage(path = qdrant_path) if not vector_storage else vector_storage
    
    def index_directory(self, skip_existing: bool = True):
        files = self.loader.get_all_files()
        print(f"Found {len(files)} files to process in directory: {self.loader.base_directory}")
        self.stats = {
            'processed_files': 0,
            'skipped_files': 0, 
            'failed_files': 0,
            'total_chunks': 0,
            'total_files': len(files)
        }
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, i, file_path, len(files), skip_existing): file_path
                for i, file_path in enumerate(files)
            }
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    with self.process_lock:
                        if result['status'] == 'processed':
                            self.stats['processed_files'] += 1
                            self.stats['total_chunks'] += result['chunks']
                        elif result['status'] == 'skipped':
                            self.stats['skipped_files'] += 1
                        elif result['status'] == 'failed':
                            self.stats['failed_files'] += 1
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
                    with self.process_lock:
                        self.stats['failed_files'] += 1
        self._print_summary()
        return self.stats
    
    def _process_single_file(self, idx: int, file_path: Union[Path, str], total_files: int, skip_existing: bool):
        thread_name = threading.current_thread().name
        try:
            print(f"\n[{idx + 1}/{total_files}] Processing: {file_path.name} (Thread: {thread_name})")
            file_hash = self.loader.compute_hash(str(file_path))

            if skip_existing and self.vector_storage.file_exists(file_hash):
                print(f"Skipping {file_path.name}, already indexed.")
                return {'status': 'skipped', 'chunks': 0}

            elements = self.loader.extract_text_from_file(str(file_path))
            if not elements:
                print(f"No text extracted from {file_path.name}, skipping.")
                return {'status': 'failed', 'chunks': 0}
            
            total_text_length = len(elements)
            print(f"Extracted {total_text_length} characters from {file_path.name}")

            chunks = self.chunker.chunk_text(elements, file_path)
            if not chunks:
                print(f"No chunks created for {file_path.name}, skipping.")
                return {'status': 'failed', 'chunks': 0}

            print(f"Created {len(chunks)} chunks")

            batch = 25
            chunk_batches = [chunks[i:i + batch] for i in range(0, len(chunks), batch)]
            total_processed = 0

            for batch_idx, chunk_batch in enumerate(chunk_batches):
                print(f"Batch {batch_idx + 1}/{len(chunk_batches)}: Generating embeddings for {len(chunk_batch)} chunks...")
                embeddings = self.embedding.generate_embedding(chunk_batch)
                self.vector_storage.store_chunks(chunk_batch, embeddings)
                total_processed += len(chunk_batch)
            
            print(f"Completed processing {file_path.name}, Total Chunks Indexed: {total_processed}")
            return {'status': 'processed', 'chunks': total_processed}
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            return {'status': 'failed', 'chunks': 0}

    
    def _print_summary(self):
        print(f"\n{'='*60}")
        print(f"  Indexing Complete!")
        print(f"   Files processed: {self.stats['processed_files']}")
        print(f"   Files skipped: {self.stats['skipped_files']}")
        print(f"   Files failed: {self.stats['failed_files']}")
        print(f"   Total chunks: {self.stats['total_chunks']:,}")
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