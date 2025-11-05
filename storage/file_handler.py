from typing import List, Optional
from pathlib import Path
import hashlib
from unstructured.partition.auto import partition

class FileHandler:

    allowed_files = {
        '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt', '.md', '.rtf', '.odt'
    }

    def __init__(self, base_directory: str):
        self.base_directory = Path(base_directory)
    
    def get_all_files(self) -> List[Path]:
        files = []
        for extension in self.allowed_files:
            files.extend(self.base_directory.rglob(f'*{extension}'))
        print(f"Found {len(files)} files with allowed extensions.")
        return files
    
    def extract_text_from_file(self, file_path: str) -> Optional[str]:
        try:
            contents = partition(filename=file_path)
            text = "\n\n".join([str(el) for el in contents])
            return text
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    @staticmethod
    def compute_hash(file_path: str):
        file_path = Path(file_path)
        hasher = hashlib.blake2b(digest_size=16)
        stat = file_path.stat()
        hash_input = f"{stat.st_size}:{stat.st_mtime}".encode()
        hasher.update(hash_input)
        return hasher.hexdigest()