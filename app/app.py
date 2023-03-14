from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, WebSocket

from dataload.dataloading import DataFilesRegistry
from pipelines import persistence
from pipelines.filtering import FilterPipeline
from pipelines.impl.anomaly_detection import _load_paragraphs
from transcribe.transcribe import AudioTranscriber
from sentence_transformers import util

CONFIG_FILE = "app.yaml"


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


app = FastAPI()
config = read_yaml(CONFIG_FILE)

MODEL_DIR = Path(config["model_dir"])
API_KEY_ASSEMBLYAI = config["api_key_assemblyai"]

filter_pipeline: FilterPipeline = persistence.load_pipeline(MODEL_DIR)
dataset_dir = Path(Path.cwd() / "../datasets")
registry = DataFilesRegistry(dataset_dir)

paragraphs = _load_paragraphs(filter_pipeline.datasets.train_id, registry)
sentences = filter_pipeline.sentence_splitter.transform(paragraphs)
embeddings = filter_pipeline.embedder.transform(sentences)


def most_similar(my_sentence, top_n=3):
    emb = filter_pipeline.embedder.transform(my_sentence)
    cosines = util.cos_sim(emb, embeddings)
    idx = np.argmax(cosines[0])
    return sentences[idx]


@app.websocket("/ws")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def callback(text):
        print("Transcribed text received:", text)
        for r in filter_pipeline.filter_sentences(text):
            await websocket.send_json(
                {
                    "relevant": bool(r.ood_proba < 0.5),
                    "sentence": r.sentence,
                    "score": round(100.00 * float(1 - r.ood_proba)),
                }
            )
            if r.ood_proba < 0.25:
                print("Relevant sentence:", r.sentence)
                print("Most similar sentences:")
                print(most_similar(r.sentence))

    await websocket.receive_text()

    audio_transcriber = AudioTranscriber(API_KEY_ASSEMBLYAI)
    audio_transcriber.subscribe(callback)
    await audio_transcriber.start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
