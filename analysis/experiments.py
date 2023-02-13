import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat
from textwrap import indent
from typing import Iterable

from pipelines.persistence import get_pipeline_info, PipelineInfoCard


@dataclass(order=True)
class Experiment:
    sort_index: datetime = field(init=False, repr=False)

    name: str
    pipeline_info: PipelineInfoCard | None
    success: bool
    f1_score: float | None

    def __post_init__(self):
        self.sort_index = datetime.strptime(self.name, "%Y-%m-%d %H-%M-%S")

    def __str__(self):
        output = io.StringIO()
        print(f"===== {self.name} =====", file=output)
        status = "SUCCESS" if self.success else "FAILED"
        print(f"Status: {status}", file=output)
        print(f"F1-Score: {self.f1_score}", file=output)
        if self.pipeline_info:
            print("Info:", file=output)
            formatted_info = pformat(self.pipeline_info)
            print(indent(formatted_info, "\t"), file=output)
        return output.getvalue()


class Experiments:
    outputs_dir: Path

    def __init__(self, outputs_dir: Path | str):
        self.outputs_dir = (
            outputs_dir if isinstance(outputs_dir, Path) else Path(outputs_dir)
        )

    def list(self) -> Iterable[Experiment]:
        for day_folder in self.outputs_dir.iterdir():
            for time_folder in day_folder.iterdir():
                yield self._load_experiment(day_folder, time_folder)

    def _load_experiment(self, day_folder, time_folder):
        experiment_instant = self._experiment_name(day_folder, time_folder)
        has_model = (time_folder / "pipeline.yml").exists()
        has_weights = (time_folder / "weights.yml").exists()
        info = get_pipeline_info(time_folder) if has_model else None
        f1_score = self._find_f1_score(time_folder / "training.log")
        e = Experiment(experiment_instant, info, has_weights, f1_score)
        return e

    @staticmethod
    def _experiment_name(day_folder, time_folder):
        return day_folder.stem + " " + time_folder.stem

    @staticmethod
    def _find_f1_score(log_file: Path) -> float | None:
        if not log_file.exists():
            return None

        with open(log_file) as f:
            lines = f.readlines()

            for line in lines:
                match = re.search(r"F1-Score=(\d+.\d+)", line)
                if match:
                    return float(match.group(1))

    def find(self, name: str) -> Experiment:
        for day_folder in self.outputs_dir.iterdir():
            for time_folder in day_folder.iterdir():
                if self._experiment_name(day_folder, time_folder) == name:
                    return self._load_experiment(day_folder, time_folder)
        raise ValueError(f"Experiment {name} not found")

    def find_folder(self, name: str) -> Path:
        for day_folder in self.outputs_dir.iterdir():
            for time_folder in day_folder.iterdir():
                if self._experiment_name(day_folder, time_folder) == name:
                    return time_folder


if __name__ == "__main__":
    experiments = Experiments(Path("../train/outputs"))
    for experiment in sorted(experiments.list()):
        print(experiment)
