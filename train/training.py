import logging
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from dataload.dataloading import DataFilesRegistry
from pipelines import persistence
from pipelines.impl.anomaly_detection import GaussianEmbeddingsAnomalyDetector


@hydra.main(config_path=".", config_name="config")
def train(cfg: DictConfig):
    """Train the filter pipeline. Pipeline train parameters are loaded from pipeline.yml.
    Thanks to hydra, each run produces a new directory with a log of the results, a copy of the pipeline.yml file
    and the trained model weights.

    Once you are happy with the results, you can copy the output directory and promote as a the "pipeline" trained
    model for the application.
    """
    working_dir = os.getcwd()
    original_config_dir = get_original_cwd()
    logging.info(f"Current working directory : {working_dir}")
    logging.info(f"Used model parameters from : {original_config_dir}")

    pipeline = persistence.load_pipeline(
        original_config_dir, GaussianEmbeddingsAnomalyDetector
    )
    registry = DataFilesRegistry(cfg.datafiles_root)
    pipeline.train(registry)

    persistence.save_pipeline(pipeline)


if __name__ == "__main__":
    train()
