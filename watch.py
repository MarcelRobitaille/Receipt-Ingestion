"""
I would have liked to use `asyncinotify` or `minotaur`, but those don't give the
full path.
"""

from typing import Union
import logging
import re
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemMovedEvent, \
    FileSystemEventHandler

from constants import env
from process_image import process_image

WATCH_DIR = Path(env('WATCH_DIR'))


class EventHandler(FileSystemEventHandler):
    def on_any_event(self, event: Union[FileSystemEvent, FileSystemMovedEvent]):
        if event.is_directory:
            return
        path = Path(event.dest_path if isinstance(event, FileSystemMovedEvent)
                    else event.src_path)

        if not re.match(r'IMG_\d{8}_\d{6}\.jpg', path.name):
            return

        try:
            process_image(path)
        except Exception as e:
            print(e)


def setup_watch():
    assert WATCH_DIR.exists()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    observer = Observer()
    observer.schedule(EventHandler(), WATCH_DIR, recursive=True)
    observer.start()
    try:
        while observer.is_alive():
            observer.join(1)
    finally:
        observer.stop()
        observer.join()
