from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from analysis.experiments import Experiments
from dataload.dataloading import DataFilesRegistry
from pipelines.impl.anomaly_detection import GaussianEmbeddingsAnomalyDetector
from pipelines.persistence import load_pipeline


def plot_cutoff_histogram(experiment_folder: Path):
    import matplotlib.pyplot as plt

    data_registry = DataFilesRegistry(
        Path("/Users/jlinho/Desktop/capstone/datasources")
    )

    detector: GaussianEmbeddingsAnomalyDetector = load_pipeline(
        experiment_folder, GaussianEmbeddingsAnomalyDetector
    )

    from pipelines.impl.anomaly_detection import _file_sentences

    test_id_sentences = list(
        _file_sentences(detector.files.validation_id, data_registry)
    )
    test_ood_sentences = list(
        _file_sentences(detector.files.validation_ood, data_registry)
    )

    # obtained scores
    id_scores = np.array([detector.mahalanobis(s) for s in test_id_sentences])
    ood_scores = np.array([detector.mahalanobis(s) for s in test_ood_sentences])

    df_id = pd.DataFrame({"score": id_scores, "origin": "ID"})
    df_ood = pd.DataFrame({"score": ood_scores, "origin": "OOD"})
    df = pd.concat([df_id, df_ood])
    sns.set_style("darkgrid")
    graph = sns.histplot(
        data=df,
        x="score",
        hue="origin",
        legend=True,
        stat="count"
    )
    graph.axvline(detector.cutoff_distance, color="grey", linestyle="--", label="Cutoff")

    plt.savefig(experiment_folder / "cutoff_val_histogram.png")


if __name__ == "__main__":
    experiments = Experiments(Path("../train/outputs"))
    experiment_folder = experiments.find_folder("2023-02-13 00-03-32")
    plot_cutoff_histogram(experiment_folder)
