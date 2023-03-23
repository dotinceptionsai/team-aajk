import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Collection
from urllib.parse import urlparse, unquote

import mlflow
import pandas as pd
from mlflow.entities import Experiment, Run, FileInfo


# create a dataclass with all the infos of a run
@dataclass
class RunInfo:
    id: str
    status: str
    start_time: datetime
    experiment_name: str
    params: dict
    metrics: dict
    artifacts: list[str]
    artifact_dir: str


class ExperimentRegistry:
    def __init__(self, mlruns_location: str | Path = "train/mlruns"):
        mlruns_path = (
            Path(mlruns_location)
            if isinstance(mlruns_location, str)
            else mlruns_location
        )
        mlruns_uri = "file://" + str(mlruns_path.absolute())
        mlflow.set_tracking_uri(mlruns_uri)
        self.tracking_uri = mlruns_uri
        self.client = mlflow.tracking.client.MlflowClient()

    def get_run_info(self, experiment_name: str, run_name: str) -> RunInfo:
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise KeyError("Experiment not found")
        for run in self.client.search_runs(experiment_ids=[experiment.experiment_id]):
            if run.info.run_name == run_name:
                artifacts = self.client.list_artifacts(run.info.run_id)
                return _build_row(experiment, run, artifacts)
        raise KeyError(f"Run {run_name} not found in experiment {experiment_name}")

    def get_all_runs(
        self, experiment_name: str, as_df: bool | None = True
    ) -> pd.DataFrame | list[RunInfo]:
        experiment: Experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise Exception("Experiment not found")
        rows = []
        run: Run
        for run in self.client.search_runs(experiment_ids=[experiment.experiment_id]):
            if run.info.status == "FINISHED" and run.info.lifecycle_stage == "active":
                artifacts = self.client.list_artifacts(run.info.run_id)
                row = _build_row(experiment, run, artifacts)
                rows.append(row)
        return _run_info_to_df(rows) if as_df else rows

    def build_archive(self, experiment_name: str, run_name: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            r = self.get_run_info(experiment_name, run_name)
            return bundle_artifacts(r.artifact_dir, str(tmp_file.name))


def _build_row(experiment: Experiment, run: Run, artifacts: Collection[FileInfo]):
    experiment_name = experiment.name
    start_time = datetime.fromtimestamp(run.info.start_time / 1000)
    params = run.data.params
    metrics = run.data.metrics
    artifact_path = Path(run.info.artifact_uri)
    return RunInfo(
        run.info.run_name,
        run.info.status,
        start_time,
        experiment_name,
        params,
        metrics,
        [str(artifact_path / a.path) for a in artifacts],
        str(artifact_path),
    )


def _run_info_to_df(rows: list[RunInfo]) -> pd.DataFrame:
    all_rows_df = pd.DataFrame()
    for r in rows:
        row_dict = {
            "id": r.id,
            "status": r.status,
            "start_time": r.start_time,
            "experiment_name": r.experiment_name,
            "artifacts": r.artifacts,
        }
        row_dict = {**row_dict, **r.params, **r.metrics}
        df = pd.DataFrame.from_dict(row_dict, orient="index").T
        all_rows_df = pd.concat([all_rows_df, df], axis=0)

    # sort by comparison_score if not empty
    if "comparison_score" in all_rows_df.columns:
        all_rows_df = all_rows_df.sort_values(by="comparison_score", ascending=False)
        all_rows_df.index = range(0, len(all_rows_df))

    return all_rows_df


def bundle_artifacts(artifact_dir: str, tar_filename: str):
    import shutil

    return shutil.make_archive(tar_filename, "gztar", _path_from_file_uri(artifact_dir))


def _path_from_file_uri(file_uri: str) -> Path:
    file_path = urlparse(file_uri).path
    decoded_file_path = unquote(file_path)
    return Path(decoded_file_path)


def fix_artifact_paths(path: str) -> None:
    """
    Because of a bug in MLFlow https://github.com/mlflow/mlflow/issues/2816 we fix the paths of the artifacts.
    Ensures that the paths of all locally saved MLflow artifacts are fixed to display them on the current machine.
    """
    for meta_yaml in Path(path).rglob("meta.yaml"):
        with open(meta_yaml.absolute()) as meta_yaml_file:
            content = meta_yaml_file.readlines()
            if "file://" not in content[0]:
                print(
                    f"[bold yellow]Skipping path fixing for: {meta_yaml.absolute()}. Run was not saved locally."
                )
        print(f"[bold blue]Fixing path for: {meta_yaml.absolute()}")
        with open(meta_yaml.absolute()) as meta_yaml_file:
            content = meta_yaml_file.readlines()
            if "artifact_location" in content[0]:
                content[
                    0
                ] = f"artifact_location: file://{meta_yaml.absolute().__str__()[:-10]}\n"
            else:
                content[
                    0
                ] = f"artifact_uri: file://{meta_yaml.absolute().__str__()[:-10]}/artifacts\n"

        print(f"File {meta_yaml.absolute()} will be set to {content[0]}")
        with open(meta_yaml.absolute(), "w") as meta_yaml_file:
            meta_yaml_file.writelines(content)


if __name__ == "__main__":
    experiments = ExperimentRegistry("/Users/jlinho/MyGit/team-aajk/train/mlruns")
    r = experiments.get_run_info(
        experiment_name="europcar", run_name="marvelous-wren-423"
    )
    ret = experiments.build_archive("europcar", "marvelous-wren-423")
    print(ret)
