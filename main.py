from fastapi import FastAPI
from fastapi.responses import FileResponse
import torchaudio as ta
import torch
import uuid
import os
from pydantic import BaseModel

# Use multilingual model
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = FastAPI()

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    language: str = "hi"  # default to Hindi

@app.post("/tts")
def tts(request: TTSRequest):
    # Specify language
    wav = model.generate(request.text, language_id=request.language)
    filename = f"{uuid.uuid4()}.wav"
    path = os.path.join(OUTPUT_DIR, filename)
    ta.save(path, wav, model.sr)
    return FileResponse(path, media_type="audio/wav", filename=filename)

from fastapi.staticfiles import StaticFiles

# Serve index.html and static files
app.mount("/", StaticFiles(directory=".", html=True), name="static")
