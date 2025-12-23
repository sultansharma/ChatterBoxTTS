from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torchaudio as ta
import torch
import os
import uuid

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = FastAPI()

# Load multilingual TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)

# Output folder
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Request model
class TTSRequest(BaseModel):
    text: str
    language: str = "hi"  # default Hindi

def synthesize_long_text(text: str, language: str):
    # Split text into sentences by "।" or newline
    sentences = [s.strip() for s in text.replace("\n","।").split("।") if s.strip()]
    chunks = []

    for sentence in sentences:
        wav = model.generate(sentence, language_id=language)
        chunks.append(wav)

    # Concatenate all audio chunks
    if len(chunks) == 1:
        final_wav = chunks[0]
    else:
        final_wav = torch.cat(chunks)

    filename = f"{uuid.uuid4()}.wav"
    path = os.path.join(OUTPUT_DIR, filename)
    ta.save(path, final_wav, model.sr)
    return path

@app.post("/tts")
def tts(request: TTSRequest):
    path = synthesize_long_text(request.text, request.language)
    return FileResponse(path, media_type="audio/wav", filename=os.path.basename(path))

# Serve static files (index.html)
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory=".", html=True), name="static")
