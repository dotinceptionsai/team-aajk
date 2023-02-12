import random
from pathlib import Path
from typing import Iterable, Collection, Optional, Callable

import yaml

from dataload import dataloading as dl
from pipelines.impl.tokenizing import to_sentences

CHATTERBOX_DIR = Path("../resources/chatterbox")
RAW_SCRAPING_DIR = Path("../resources/scraping")


def load_chatterbox(
    path: Path | str, restrict_to: Optional[Collection[str]] = None
) -> dict[str, list[str]]:
    if isinstance(path, str):
        path = Path(path)
    keep_cond = (
        (lambda _: True)
        if restrict_to is None
        else (lambda filename: filename in restrict_to)
    )
    return {
        name: sentences for name, sentences in _load_chatterbox_gen(path, keep_cond)
    }


def _load_chatterbox_gen(
    path: Path, keep_cond: Callable[[str], bool]
) -> Iterable[tuple[str, list[str]]]:
    for p in path.glob("**/*.yml"):
        if keep_cond(p.stem):
            yield p.stem, _parse_chatterbox(p)


def _parse_chatterbox(p: Path) -> list[str]:
    data = yaml.safe_load(p.read_text())
    return list(_parse_chatterbox_rec(data["conversations"]))


def _parse_chatterbox_rec(data) -> Iterable[str]:
    if isinstance(data, str):
        yield data.strip()
    else:
        for d in data:
            yield from _parse_chatterbox_rec(d)


def generate_id_sample(domain_name: str) -> list[str]:
    loader = dl.DataFilesRegistry(RAW_SCRAPING_DIR)
    id_paragraphs = loader.load_items(domain_name)

    questions = [para for idx, para in enumerate(id_paragraphs) if idx % 2 == 0]
    answers = [para for idx, para in enumerate(id_paragraphs) if idx % 2 == 1]

    questions_sentences = [s for q in questions for s in to_sentences(q)]
    answers_sentences = [s for q in answers for s in to_sentences(q)]

    sample_questions = random.sample(questions_sentences, 20)
    sample_answer = random.sample(answers_sentences, 60)
    return sample_questions + sample_answer


def generate_ood_sample() -> list[str]:
    sampling_dist: dict[str, int] = {
        "conversations": 20,
        "greetings": 20,
        "psychology": 20,
        "support": 20,
    }
    chatterbox = load_chatterbox(CHATTERBOX_DIR, sampling_dist.keys())
    ood_samples = []
    for chatter_name, sentences in chatterbox.items():
        file_samples = random.sample(sentences, sampling_dist[chatter_name])
        ood_samples.extend(file_samples)
    return ood_samples


def generate_and_export_samples(target_dir: Path | str, domain_name: str):
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    id_samples = generate_id_sample(domain_name)
    ood_samples = generate_ood_sample()

    with open(target_dir / f"validation_{domain_name}_id.yml", "w") as f:
        yaml.dump(ood_samples, f, sort_keys=True)

    with open(target_dir / f"validation_{domain_name}_id.yml", "w") as f:
        yaml.dump(id_samples, f, sort_keys=True)


if __name__ == "__main__":
    for f in Path(RAW_SCRAPING_DIR).glob("*.yml"):
        print(f.stem)
        generate_and_export_samples(
            "/Users/jlinho/Desktop/capstone/datasources", f.stem
        )
