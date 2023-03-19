from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Collection

import mlflow
import pandas as pd
import tarfile
from urllib.parse import urlparse, unquote
from pathlib import Path


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


def get_all_runs(experiment_name: str, as_df: bool | None = True) -> list[RunInfo]:
    client = mlflow.tracking.client.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception("Experiment not found")
    rows = []
    for run in client.search_runs(experiment_ids=[experiment.experiment_id]):
        if run.info.status == "FINISHED" and run.info.lifecycle_stage == "active":
            row = build_row(client, experiment, run)
            rows.append(row)
    return to_dataframe(rows) if as_df else rows


def build_row(client, experiment, run):
    experiment_name = experiment.name
    start_time = datetime.fromtimestamp(run.info.start_time / 1000)
    params = run.data.params
    metrics = run.data.metrics
    artifacts = client.list_artifacts(run.info.run_id)
    artifact_path = Path(run.info.artifact_uri)
    artifacts = [str(artifact_path / a.path) for a in artifacts]
    # Put data into a dataclass of type RunInfo
    row = RunInfo(
        run.info.run_name,
        run.info.status,
        start_time,
        experiment_name,
        params,
        metrics,
        artifacts,
    )
    return row


def to_dataframe(rows: list[RunInfo]) -> pd.DataFrame:
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
        # concatenate the three dataframes
        df = pd.DataFrame.from_dict(row_dict, orient="index").T
        all_rows_df = pd.concat([all_rows_df, df], axis=0)

    # sort by comparison_score if not empty
    if "comparison_score" in all_rows_df.columns:
        all_rows_df = all_rows_df.sort_values(by="comparison_score", ascending=False)
        all_rows_df.index = range(0, len(all_rows_df))

    return all_rows_df


def bundle_artifacts(artifacts: Collection[str], tar_filename: str):
    tar_file = tarfile.open(tar_filename, "w:gz")

    for artifact in artifacts:
        path = path_from_file_uri(artifact)
        if path.exists() and path.is_file():
            tar_file.add(str(path), arcname=path.name)

    tar_file.close()


def path_from_file_uri(file_uri: str) -> Path:
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
    mlruns = str(str(Path("/tmp/train/mlruns").absolute()))
    mlflow.set_tracking_uri("file://" + mlruns)
    fix_artifact_paths(mlruns)

    rows = get_all_runs("europcar")

    # get the best run
    # find first row with "id" equals "marvelous-wren-423"
    row = rows[rows["id"] == "marvelous-wren-423"].iloc[0]
    zipfile = bundle_artifacts(row.artifacts, "artifacts.tar.gz")
    print(zipfile)

    # find experiment by name
    rows.to_csv("all_runs.csv", index=True)
