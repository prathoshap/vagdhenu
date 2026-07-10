# src/api/chant.py

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from core.audio import generate_audio
import tempfile
import os

router = APIRouter()

class ChantRequest(BaseModel):
    sloka: str

@router.post("/chant")
def chant_endpoint(req: ChantRequest):
    # Create a temporary MP3 file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp_file.name
    tmp_file.close()

    # Call your existing audio generator
    generate_audio(req.sloka, tmp_path)

    # Return MP3 file
    return FileResponse(
        tmp_path,
        media_type="audio/mpeg",
        filename="chant.mp3"
    )
