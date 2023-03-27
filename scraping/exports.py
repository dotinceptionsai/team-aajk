""" Functions to export scraped data to yml files (for ML parsing) or TSV (for nice viz on GitHub and excel)."""
from pathlib import Path
from typing import Iterable, Collection

import pandas as pd
import yaml

from scraping.scrapers import Scraper, QA


def export_yml(scraper_name: str, qas: Iterable[QA], target_folder: Path):
    filename = scraper_name + ".yml"
    yml_file = target_folder / filename
    with open(yml_file, "w") as f:
        items = []
        for i in qas:
            items.append(i.question)
            items.append(i.answer)
        yaml.dump(items, f, sort_keys=False)


def export_tsv(scraper_name: str, qas: Collection[QA], target_folder: Path):
    filename = scraper_name + ".tsv"
    csv_file = target_folder / filename
    questions = [qa.question for qa in qas]
    answers = [qa.answer.replace('"', "") for qa in qas]
    df = pd.DataFrame({"question": questions, "answer": answers})
    df.to_csv(csv_file, sep="\t", index=False, header=True)


if __name__ == "__main__":
    target_dir = Path("../resources/scraping")
    scrapers = Scraper.all()
    for name, scraper in scrapers.items():
        scraped = scraper.scrape()
        export_yml(name, scraped, target_dir)
        export_tsv(name, scraped, target_dir)
