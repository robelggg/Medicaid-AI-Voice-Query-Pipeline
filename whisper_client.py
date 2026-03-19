import os
from openai import OpenAI

SUPPORTED_EXTENSIONS = {".flac", ".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".webm"}


def transcribe(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _, ext = os.path.splitext(audio_path)
    if ext.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
    return result.text.strip()
