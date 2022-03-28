from datetime import datetime
import base64
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel

from process_image import process_image, ProcessImageResult

app = FastAPI()
app.mount('/static', StaticFiles(directory='./src'), name='static')


class Thing(BaseModel):
    image: str


@app.get('/')
def index():
    return FileResponse('./src/index.html')


@app.post('/process', response_model=ProcessImageResult)
def process(item: Thing):
    out = Path('./out')
    out.mkdir(parents=True, exist_ok=True)
    filename = out / datetime.now().strftime('%Y-%m-%d_%H:%M:%S.png')
    with open(filename, 'wb') as f:
        f.write(base64.b64decode(item.image.split(',', 1)[1]))
    res = process_image(filename)
    return res
