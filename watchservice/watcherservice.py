import asyncio
import time
import threading
from watchdog.observers import Observer
from .watcher import FolderWatcher, WatchEvent, Event
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from app.core.query_service import QueryService

from app.config import settings
from storage.file_handler import FileHandler
from app.models.requests import IngestRequest


logger = logging.getLogger(__name__)

class FolderWatcherService:
    def __init__(self, query_service: "QueryService", 
                 batch_size: int = settings.WATCHER_BATCH_SIZE,
                 poll_interval: float = settings.WATCHER_POLL_INTERVAL,
                 ):
        self.query_service = query_service
        self.batch_size = batch_size
        self.poll_interval = poll_interval

        self.event_queue = Queue()
        self.observer = Observer()
        self.watch_folders: Dict[str, Dict] = {}

        self.is_running = False
        self.process_thread = None

        self.pending_events: Dict[str, WatchEvent] = {}
        self.last_processed_time = time.time()

    def add_watch_folder(self, folder_path: str, allowed_extensions: List[str] = None, recursive: bool = True, auto_ingest: bool = True) -> bool:
        try:
            path = Path(folder_path)
            if not path.exists() or not path.is_dir():
                logger.error(f"Invalid folder path: {folder_path}")
                return False
            
            if folder_path in self.watch_folders:
                logger.error(f"Folder already being watched: {folder_path}")
                return False  

            event_handler = FolderWatcher(
                event_queue=self.event_queue,
                allowed_extensions=allowed_extensions
            )
            watch = self.observer.schedule(event_handler, str(folder_path), recursive=recursive)
            self.watch_folders[str(folder_path)] = {
                "path": str(folder_path),
                "watch": watch,
                "handler": event_handler,
                "allowed_extensions": allowed_extensions,
                "recursive": recursive,
                "auto_ingest": auto_ingest,
                "added_at": time.time()
            }
            logger.info(f"Started watching folder: {folder_path}")
            return True
        except Exception as e:
            logger.error(f"Error adding watch folder {folder_path}: {e}")
            return False
    
    def remove_watch_folder(self, folder_path: str) -> bool:
        try:
            if folder_path in self.watch_folders:
                watch_info = self.watch_folders[folder_path]
                self.observer.unschedule(watch_info['watch'])
                del self.watch_folders[folder_path]
                logger.info(f"Removed watch folder: {folder_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove watch folder {folder_path}: {e}")
            return False

    def start_watching(self):
        if self.is_running:
            logger.warning("Folder watcher is already running")
            return
        
        try:
            self.is_running = True
            self.observer.start()
            self.process_thread = threading.Thread(
                target=self._process_events,
                daemon=True
            )
            self.process_thread.start()
            logger.info("Folder watcher service started")
        except Exception as e:
            logger.error(f"Failed to start folder watcher: {e}")
            self.is_running = False
    
    def stop_watching(self):
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            self.observer.stop()
            self.observer.join(timeout=5)
            if self.process_thread and self.process_thread.is_alive():
                self.process_thread.join(timeout=5)
            logger.info("Folder watcher service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping folder watcher: {e}")
    
    def _process_events(self):
        while self.is_running:
            try:
                events_to_process = []
                try:
                    event = self.event_queue.get(timeout=1.0)
                    events_to_process.append(event)
                    while len(events_to_process) < self.batch_size:
                        try:
                            event = self.event_queue.get_nowait()
                            events_to_process.append(event)
                        except:
                            break
                except:
                    continue
                if events_to_process:
                    self._handle_event_batch(events_to_process)
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                time.sleep(1)
    
    def _handle_event_batch(self, events: List[WatchEvent]):
        try:
            files_to_ingest = set()
            files_to_delete = set()
            
            for event in events:
                file_path = str(event.src_path)

                if event.event_type in [Event.CREATED, Event.MODIFIED]:
                    if self._file_changed(event.src_path):
                        files_to_ingest.add(file_path)
                        
                elif event.event_type == Event.DELETED:
                    files_to_delete.add(file_path)
                        
                elif event.event_type == Event.MOVED:
                    if event.dest_path:
                        files_to_delete.add(str(event.dest_path))
                    files_to_ingest.add(file_path)
            
            if files_to_delete:
                self._handle_file_deletions(files_to_delete)
            
            if files_to_ingest:
                self._handle_file_ingestions(files_to_ingest)
                
        except Exception as e:
            logger.error(f"Error handling event batch: {e}")
    
    def _file_changed(self, file_path: Path) -> bool:
        try:
            if not file_path.exists():
                return False
            
            current_hash = FileHandler.compute_hash(str(file_path))
            
            return not self.query_service.vector_storage.file_exists(current_hash)
            
        except Exception as e:
            logger.error(f"Error checking file change {file_path}: {e}")
            return True 
    
    def _handle_file_ingestions(self, file_paths: Set[str]):
        try:
            for file_path in file_paths:
                folder_config = self._find_folder_config(file_path)
                if folder_config and folder_config.get('auto_ingest', True):
                    file_hash = FileHandler.compute_hash(file_path)
                    if not self.query_service.vector_storage.file_exists(file_hash):
                        self._ingest_single_file(file_path, folder_config)
                    else:
                        logger.info(f"File unchanged, skipping: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error handling file ingestions: {e}")
    
    def _handle_file_deletions(self, file_paths: Set[str]):
        try:
            for file_path in file_paths:
                self._delete_file_from_vector_store(file_path)
                
        except Exception as e:
            logger.error(f"Error handling file deletions: {e}")
    
    def _find_folder_config(self, file_path: str) -> Optional[Dict]:
        file_path = Path(file_path)
        
        for folder_path, config in self.watch_folders.items():
            folder_path = Path(folder_path)
            try:
                file_path.relative_to(folder_path)
                return config
            except ValueError:
                continue
        
        return None
    
    def _ingest_single_file(self, file_path: str, folder_config: Dict):
        try:
            file_path = Path(file_path)
            parent_dir = str(file_path.parent)
            
            request = IngestRequest(
                directory_path=parent_dir,
                skip_existing=True,  
                chunk_size=settings.default_chunk_size,
                overlap=settings.default_overlap,
                max_workers=1  
            )
            result = self.query_service.ingest(request)
            
            logger.info(f"Auto-ingested file: {file_path} - "
                       f"Chunks: {result.total_chunks}, "
                       f"Time: {result.processing_time_ms}ms")
            
        except Exception as e:
            logger.error(f"Error auto-ingesting file {file_path}: {e}")
    
    def _delete_file_from_vector_store(self, file_path: str):
        try:
            logger.info(f"File deleted (vector store cleanup needed): {file_path}")
            # TODO: Implement vector store deletion by file path
            
        except Exception as e:
            logger.error(f"Error deleting file from vector store {file_path}: {e}")
    
    def get_watch_status(self) -> Dict[str, Any]:
        return {
            'is_running': self.is_running,
            'watched_folders': list(self.watch_folders.keys()),
            'total_watched_folders': len(self.watch_folders),
            'queue_size': self.event_queue.qsize(),
            'observer_is_alive': self.observer.is_alive() if hasattr(self.observer, 'is_alive') else False
        }
    
    def get_watched_folders(self) -> List[Dict[str, Any]]:
        folders = []
        for path, config in self.watch_folders.items():
            folders.append({
                'path': path,
                'allowed_extensions': config.get('allowed_extensions', []),
                'recursive': config['recursive'],
                'auto_ingest': config['auto_ingest'],
                'added_at': config['added_at']
            })
        return folders
    