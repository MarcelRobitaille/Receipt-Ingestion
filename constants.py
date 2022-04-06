from pathlib import Path

from environs import Env


env = Env(expand_vars=True)
env.read_env()

# TODO: Determine this based on log level or something
DEBUG = True
DEBUG_IMAGE_DIR = Path('/tmp/receipt-ingestion')
DEBUG_IMAGE_DIR.mkdir(exist_ok=True)
