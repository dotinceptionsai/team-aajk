import logging
import os
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from hydra.utils import get_original_cwd
from matplotlib import pyplot as plt
from mlflow import log_metric, log_param, log_artifacts
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from analysis.evaluate import evaluate_model, evaluate_speed_ms
from dataload.dataloading import DataFilesRegistry
from pipelines import persistence

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Train the filter pipeline. Pipeline train parameters are loaded from pipeline.yml.
    Thanks to hydra, each run produces a new directory with a log of the results, a copy of the pipeline.yml file
    and the trained model weights.

    Once you are happy with the results, you can copy the output directory and promote as a the "pipeline" trained
    model for the application.
    """

    output_dir = Path(os.getcwd())
    datafiles_dir = Path(get_original_cwd()) / cfg.datafiles_root
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Datafiles dir: {datafiles_dir}")

    mlflow.set_tracking_uri("file://" + get_original_cwd() + "/mlruns")
    # experiment_name = str(output_dir.relative_to(Path(get_original_cwd()) / "multirun"))
    experiment_name = ",".join(cfg.pipeline.datasets.train_id)
    mlflow.set_experiment(str(experiment_name))

    with mlflow.start_run():
        try:
            # Train pipeline
            pipeline = hydra.utils.instantiate(cfg.pipeline)
            registry = DataFilesRegistry(datafiles_dir)
            pipeline.fit(registry)
            persistence.save_pipeline(pipeline, output_dir)

            log_param("train_id", cfg.pipeline.datasets.train_id)
            log_param("val_id", cfg.pipeline.datasets.validation_id)
            log_param("val_ood", cfg.pipeline.datasets.validation_ood)
            log_param("embedder_name", cfg.pipeline.run_params.embedder_name)
            log_param("robust_covariance", cfg.pipeline.run_params.robust_covariance)
            log_param("support_fraction", cfg.pipeline.run_params.support_fraction)

            # Log metrics and graphs
            val_id_sentences = registry.load_items(cfg.pipeline.datasets.validation_id)
            val_ood_sentences = registry.load_items(
                cfg.pipeline.datasets.validation_ood
            )
            metrics = evaluate_model(pipeline, val_id_sentences, val_ood_sentences)
            val_sentences = [*val_id_sentences, *val_ood_sentences]

            f1 = round(float(metrics.f1), 2)
            log_loss = round(float(metrics.log_loss), 5)
            speed_ms = round(float(evaluate_speed_ms(pipeline, val_sentences)), 2)
            comparison_score = 100 * f1 - 50 * log_loss - speed_ms
            comparison_score = comparison_score / 100 if comparison_score > 0 else 0

            log_metric("f1", f1)
            log_metric("log_loss", log_loss)
            log_metric("speed_ms", speed_ms)
            log_metric("comparison_score", comparison_score)

            fp_sentences = pd.DataFrame(
                {"sentence": [val_id_sentences[i] for i in metrics.fp_indices]}
            )
            fn_sentences = pd.DataFrame(
                {"sentence": [val_ood_sentences[i] for i in metrics.fn_indices]}
            )

            fp_sentences.to_csv(
                output_dir / "false_positives.csv", index=False, header=False
            )
            fn_sentences.to_csv(
                output_dir / "false_negatives.csv", index=False, header=False
            )

            cm = confusion_matrix(metrics.y_true, metrics.y_pred)
            ConfusionMatrixDisplay(cm).plot()
            plt.savefig(output_dir / "confusion_matrix.png")

            pred, sent = pipeline.predict_proba([*val_id_sentences, *val_ood_sentences])
            pd.DataFrame({"sentence": sent, "OOD prob": pred}).to_csv(
                output_dir / "predictions.csv", index=False
            )
        except Exception as e:
            logger.error(e)
            raise e
        finally:
            log_artifacts(str(output_dir.absolute()))


if __name__ == "__main__":
    train()
