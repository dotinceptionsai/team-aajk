from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd


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
        rows.append(row)
    return to_dataframe(rows) if as_df else rows


def to_dataframe(rows: list[RunInfo]) -> pd.DataFrame:
    all_rows_df = pd.DataFrame()
    for r in rows:
        row_dict = {
            "id": r.id,
            "status": r.status,
            "start_time": r.start_time,
            "experiment_name": r.experiment_name,
        }
        row_dict = {**row_dict, **r.params, **r.metrics}
        # concatenate the three dataframes
        df = pd.DataFrame.from_dict(row_dict, orient="index").T
        all_rows_df = pd.concat([all_rows_df, df], axis=0)

    all_rows_df = all_rows_df.sort_values(by="comparison_score", ascending=False)
    all_rows_df.index = range(0, len(all_rows_df))

    return all_rows_df


if __name__ == "__main__":
    mlruns = str(str(Path("../train/mlruns").absolute()))
    mlflow.set_tracking_uri("file://" + mlruns)

    rows = get_all_runs("wwf")
    # find experiment by name
    rows.to_csv("all_runs.csv", index=True)
