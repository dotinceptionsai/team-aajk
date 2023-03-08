from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, log_loss


@dataclass
class Scoring:
    f1: float
    log_loss: float
    tp_indices: Iterable[int]
    fp_indices: Iterable[int]
    tn_indices: Iterable[int]
    fn_indices: Iterable[int]
    y_true: Iterable[int]
    y_pred: Iterable[int]


def evaluate_model(my_model, id_sentences, ood_sentences):
    id_probs, id_items = my_model.predict_proba(id_sentences)
    ood_probs, ood_items = my_model.predict_proba(ood_sentences)

    y_probs = np.hstack([np.array(id_probs), np.array(ood_probs)])
    y_true = np.concatenate([np.zeros(len(id_items)), np.ones(len(ood_items))])
    y_pred = (y_probs > 0.5).astype(int)

    loss = log_loss(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)

    tp_indices, *_ = np.where((y_true == 1) & (y_pred == 1))
    fp_indices, *_ = np.where((y_true == 0) & (y_pred == 1))
    tn_indices, *_ = np.where((y_true == 0) & (y_pred == 0))
    fn_indices, *_ = np.where((y_true == 1) & (y_pred == 0))

    shifted_fn_indices = [i - len(id_items) for i in fn_indices]
    shifted_tp_indices = [i - len(id_items) for i in tp_indices]

    return Scoring(
        f1,
        loss,
        shifted_tp_indices,
        fp_indices,
        tn_indices,
        shifted_fn_indices,
        y_true,
        y_pred,
    )
