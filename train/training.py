import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from analysis.evaluate import evaluate_model
from dataload.dataloading import DataFilesRegistry
from pipelines import persistence

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    """Train the filter pipeline. Pipeline train parameters are loaded from pipeline.yml.
    Thanks to hydra, each run produces a new directory with a log of the results, a copy of the pipeline.yml file
    and the trained model weights.

    Once you are happy with the results, you can copy the output directory and promote as a the "pipeline" trained
    model for the application.
    """
    working_dir = os.getcwd()
    output_dir = cfg.output_dir
    logger.info(f"Using model parameters from : {working_dir}")

    pipeline = hydra.utils.instantiate(cfg.pipeline)
    registry = DataFilesRegistry(cfg.datafiles_root)
    pipeline.fit(registry)

    persistence.save_pipeline(pipeline, output_dir)

    val_id_sentences = registry.load_items(cfg.pipeline.datasets.validation_id)
    val_ood_sentences = registry.load_items(cfg.pipeline.datasets.validation_ood)
    metrics = evaluate_model(pipeline, val_id_sentences, val_ood_sentences)

    metrics_to_save = {
        "f1": round(float(metrics.f1), 2),
        "log_loss": round(float(metrics.log_loss), 5),
    }

    logger.info(f"Metrics: {metrics_to_save}")
    out = Path(output_dir)
    metrics_file = out / "metrics.yml"
    persistence._write_yml(metrics_file, metrics_to_save)

    fp_sentences = pd.DataFrame(
        {"sentence": [val_id_sentences[i] for i in metrics.fp_indices]}
    )
    fn_sentences = pd.DataFrame(
        {"sentence": [val_ood_sentences[i] for i in metrics.fn_indices]}
    )

    fp_sentences.to_csv(out / "false_positives.csv", index=False, header=False)
    fn_sentences.to_csv(out / "false_negatives.csv", index=False, header=False)

    cm = confusion_matrix(metrics.y_true, metrics.y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(out / "confusion_matrix.png")

    pred, sent = pipeline.predict_proba([*val_id_sentences, *val_ood_sentences])
    pd.DataFrame({"sentence": sent, "OOD prob": pred}).to_csv(
        out / "predictions.csv", index=False
    )


if __name__ == "__main__":
    train()
