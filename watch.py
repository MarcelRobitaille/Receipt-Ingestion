from typing import Union
import logging
import re
from pathlib import Path

from asyncinotify import Inotify, Mask

from constants import env
from process_image import process_image

WATCH_DIR = Path(env('WATCH_DIR'))


def setup_watch():
    assert WATCH_DIR.exists()
    with Inotify() as inotify:
        for folder in glob(f'{WATCH_DIR}/**/', recursive=True):
            inotify.add_watch(Path(folder), Mask.CREATE | Mask.MOVE)
        async for event in inotify:
            if not re.match(r'IMG_\d{8}_\d{6}\.jpg', event.name.name):
                continue
            print(dir(event))
            print(type(event.name))
            print(event.name)
            process_image(event.name)
