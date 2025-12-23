from fastapi import FastAPI
from fastapi.responses import FileResponse
import torchaudio as ta
import torch
import uuid
import os

from chatterbox.tts import ChatterboxTTS

app = FastAPI()

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str
@app.post("/tts")
def tts(request: TTSRequest):
    wav = model.generate(request.text)
    filename = f"{uuid.uuid4()}.wav"
    path = os.path.join(OUTPUT_DIR, filename)
    ta.save(path, wav, model.sr)
    return FileResponse(path, media_type="audio/wav", filename=filename)

from fastapi.staticfiles import StaticFiles

#Hello Serve index.html and any other static files from this folder
app.mount("/", StaticFiles(directory=".", html=True), name="static")