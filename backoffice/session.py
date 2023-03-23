# Create an enum  class named OnInvalidSentence with values FAIL or WARN
from dataclasses import dataclass
from enum import Enum

DEFAULT_CLEANUPS = ["spaces", "sentences_start", "uri", "email", "common_abbr"]


@dataclass
class State:
    selected_dataset: str = None
    selected_base_model: str = None
    selected_calibration: str = None
    current_stage: int = 0


class SessionKey(str, Enum):
    SELECTED_DATASET = "selected_dataset"
    SELECTED_CLEANUPS = "selected_cleanups"
    SELECTED_BASE_MODEL = "selected_base_model"
    SELECTED_CALIBRATION = "selected_calibration"
    SELECTED_VERIFICATION = "selected_verification"
    SELECTED_SETUP_STAGE = "selected_setup_stage"
    SELECTED_SETUP = "selected_setup"
