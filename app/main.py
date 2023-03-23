import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict
from urllib.parse import urlparse, unquote

from deepgram import Deepgram
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi import WebSocket, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dataload.dataloading import DataFilesRegistry
from pipelines import persistence
from pipelines.filtering import FilteredSentence


class TextMessage(BaseModel):
    text: str


load_dotenv()

active_pipeline = dict()

model_dir = os.getenv("MODEL_DIR")
# filter_pipeline: FilterPipeline | None  # = persistence.load_pipeline(os.getenv("MODEL_DIR"))
# filter_pipeline_dir: Path | None = None
dg_client = Deepgram(os.getenv("DEEPGRAM_API_KEY"))

dataset_dir = Path(Path.cwd() / "../datasets")
registry = DataFilesRegistry(dataset_dir)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/ready")
async def ready():
    return {
        "ready": "pipe" in active_pipeline,
        "filter_pipeline_dir": to_model_uri(active_pipeline["dir"])
        if "dir" in active_pipeline
        else None,
    }


class ModelChoice(BaseModel):
    uri: str


@app.post("/active_model")
async def set_active_model(model: ModelChoice):
    model_path = to_model_path(model.uri)

    if not model_path.exists():
        raise Exception("Model not found")

    test_pipe = persistence.load_pipeline(model_path)
    for s in test_pipe.filter_sentences("test me"):
        print(s)
    active_pipeline["pipe"] = test_pipe
    active_pipeline["dir"] = model_path
    return model


def to_model_uri(model_path: Path):
    return f"/models/{model_path.parent.name}/{model_path.name}"


def to_model_path(model_uri):
    file_path = urlparse(model_uri).path
    decoded_file_path = unquote(file_path)
    folder_path = decoded_file_path.split("/models/")[1]
    model_path = Path(f"{model_dir}/{folder_path}")
    return model_path


@app.post("/models/{category}")
async def create_model(
    category: str,
    preload: bool = Query(
        False,
        description="Whether to run the preload action after decompressing the files.",
    ),
    file: UploadFile = File(
        ..., description="The tar.gz file containing the model files."
    ),
):
    unique_model_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = f"{model_dir}/{category}/{unique_model_name}"
    os.makedirs(folder_path, exist_ok=True)
    file_path = f"{folder_path}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    shutil.unpack_archive(file_path, folder_path)
    # delete file archive
    Path(file_path).unlink()

    if preload:
        test_pipe = persistence.load_pipeline(folder_path)
        for s in test_pipe.filter_sentences("This is a test"):
            print(s)

    return {"uri": f"/models/{category}/{unique_model_name}"}


@app.get("/models")
async def list_models():
    models = []

    model_directories = [str(path) for path in Path(model_dir).glob("*/20*")]

    for d in model_directories:
        model_uri = to_model_uri(Path(d))
        model_dict = {"model_uri": model_uri, "category": d.split("/")[-2]}
        models.append(model_dict)

    return models


@app.get("/")
async def redirect_typer():
    return RedirectResponse("/static/index.html")


@app.post("/text")
def on_text_message(message: TextMessage):
    filter_pipeline = active_pipeline["pipe"]
    if filter_pipeline is None:
        raise Exception("No model loaded")
    return [as_json_result(r) for r in filter_pipeline.filter_sentences(message.text)]


@app.websocket("/listen")
async def on_audio_message(websocket: WebSocket):
    filter_pipeline = active_pipeline["pipe"]
    if filter_pipeline is None:
        raise Exception("No model loaded")
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
    pipe_ = active_pipeline["pipe"]

    async def get_transcript(data: Dict) -> None:
        if "channel" in data:
            transcript = data["channel"]["alternatives"][0]["transcript"]

            if transcript:
                print("Transcribed text received:", transcript)

                for r in pipe_.filter_sentences(transcript):
                    await fast_socket.send_json(as_json_result(r))

    deepgram_socket = await connect_to_deepgram(get_transcript)
    return deepgram_socket


async def connect_to_deepgram(transcript_received_handler: Callable[[Dict], None]):
    try:
        socket = await dg_client.transcription.live(
            {
                "punctuate": True,
                "interim_results": False,
                "numerals": True,
            }
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

    uvicorn.run("main:app", host="0.0.0.0", port=8002)
