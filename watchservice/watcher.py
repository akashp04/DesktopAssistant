import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import logging
from enum import Enum
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class Event(Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"

@dataclass
class WatchEvent:
    event_type: Event
    src_path: Path
    timestamp: float
    dest_path: Optional[Path] = None

class FolderWatcher(FileSystemEventHandler):
    def __init__(self, event_queue: Queue,
                 allowed_extensions: Optional[List[str]] = None):
        super().__init__()
        self.event_queue = event_queue
        self.allowed_extensions = allowed_extensions
        self.ignored_file_types = {'.DS_Store', '.gitignore', '~$*', '*.tmp'}

    def _should_process_file(self, file_path: str) -> bool:
        path = Path(file_path)
        for pattern in self.ignored_file_types:
            if pattern in path.name or path.name.startswith(pattern.replace('*', '')):
                return False
        if self.allowed_extensions and path.suffix.lower() not in self.allowed_extensions:
            return False
            
        return True

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and self._should_process_file(event.src_path):
            watch_event = WatchEvent(
                event_type=Event.CREATED,
                src_path=Path(event.src_path),
                timestamp=time.time()
            )
            self.event_queue.put(watch_event)
            logger.info(f"File created: {event.src_path}")
    
    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self._should_process_file(event.src_path):
            watch_event = WatchEvent(
                event_type=Event.MODIFIED,
                src_path=Path(event.src_path),
                timestamp=time.time()
            )
            self.event_queue.put(watch_event)
            logger.info(f"File modified: {event.src_path}")
    
    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and self._should_process_file(event.src_path):
            watch_event = WatchEvent(
                event_type=Event.DELETED,
                src_path=Path(event.src_path),
                timestamp=time.time()
            )
            self.event_queue.put(watch_event)
            logger.info(f"File deleted: {event.src_path}")
    
    def on_moved(self, event: FileSystemEvent):
        if not event.is_directory and (self._should_process_file(event.src_path) or self._should_process_file(event.dest_path)):
            watch_event = WatchEvent(
                event_type=Event.MOVED,
                src_path=Path(event.src_path),
                dest_path=Path(event.dest_path),
                timestamp=time.time()
            )
            self.event_queue.put(watch_event)
            logger.info(f"File moved from {event.src_path} to {event.dest_path}")

