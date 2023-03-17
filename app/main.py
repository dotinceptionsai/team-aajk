import os
from pathlib import Path
from typing import Callable, Dict

from deepgram import Deepgram
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from dataload.dataloading import DataFilesRegistry
from pipelines import persistence
from pipelines.filtering import FilterPipeline, FilteredSentence


class TextMessage(BaseModel):
    text: str


load_dotenv()

filter_pipeline: FilterPipeline = persistence.load_pipeline(os.getenv("MODEL_DIR"))
dg_client = Deepgram(os.getenv("DEEPGRAM_API_KEY"))

dataset_dir = Path(Path.cwd() / "../datasets")
registry = DataFilesRegistry(dataset_dir)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def redirect_typer():
    return RedirectResponse("/static/index.html")


@app.post("/text")
def on_text_message(message: TextMessage):
    return [as_json_result(r) for r in filter_pipeline.filter_sentences(message.text)]


@app.websocket("/listen")
async def on_audio_message(websocket: WebSocket):
    await websocket.accept()

    try:
        deepgram_socket = await process_audio(websocket)

        while True:
            data = await websocket.receive_bytes()
            deepgram_socket.send(data)
    except Exception as e:
        raise Exception(f"Could not process audio: {e}")
    finally:
        await websocket.close()


async def process_audio(fast_socket: WebSocket):
    async def get_transcript(data: Dict) -> None:
        if "channel" in data:
            transcript = data["channel"]["alternatives"][0]["transcript"]

            if transcript:
                print("Transcribed text received:", transcript)
                for r in filter_pipeline.filter_sentences(transcript):
                    await fast_socket.send_json(as_json_result(r))

    deepgram_socket = await connect_to_deepgram(get_transcript)
    return deepgram_socket


async def connect_to_deepgram(transcript_received_handler: Callable[[Dict], None]):
    try:
        socket = await dg_client.transcription.live(
            {"punctuate": True, "interim_results": False}
        )
        socket.registerHandler(
            socket.event.CLOSE, lambda c: print(f"Connection closed with code {c}.")
        )
        socket.registerHandler(
            socket.event.TRANSCRIPT_RECEIVED, transcript_received_handler
        )

        return socket
    except Exception as e:
        raise Exception(f"Could not open socket: {e}")


def as_json_result(r: FilteredSentence):
    return {
        "relevant": bool(r.ood_proba < 0.5),
        "sentence": r.sentence,
        "score": round(100.00 * float(1 - r.ood_proba)),
    }


if __name__ == "__main__":
    import uvicorn

    for s in filter_pipeline.filter_sentences("This is a test sentence."):
        print(s)
    uvicorn.run("main:app", host="0.0.0.0", port=8002)
