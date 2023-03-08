from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket

from pipelines import persistence
from pipelines.filtering import FilterPipeline
from transcribe.transcribe import AudioTranscriber

CONFIG_FILE = "config.yml"


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


app = FastAPI()
config = read_yaml(CONFIG_FILE)

MODEL_DIR = Path(config["model_dir"])
API_KEY_ASSEMBLYAI = config["api_key_assemblyai"]

filter_pipeline: FilterPipeline = persistence.load_pipeline(MODEL_DIR)


@app.websocket("/ws")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def callback(text):
        print("Transcribed text received:", text)
        for r in filter_pipeline.filter_sentences(text):
            await websocket.send_json(
                {
                    "relevant": bool(r.relevant) and float(r.score) <= 20,
                    "sentence": r.sentence,
                    "score": 100.00 - float(r.score),
                }
            )

    await websocket.receive_text()

    audio_transcriber = AudioTranscriber(API_KEY_ASSEMBLYAI)
    audio_transcriber.subscribe(callback)
    await audio_transcriber.start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
