from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error
import numpy as np
import math


def prc_auc(targets, preds):
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)

def compute_cls_metric(y_true, y_pred):
    masked_y_true = (y_true[y_true != None]).astype(np.float32)
    masked_y_pred = (y_pred[y_true != None]).astype(np.float32)
    if np.sum(masked_y_true) > 0 and np.sum(masked_y_true) < masked_y_pred.shape[0]:
        prc = prc_auc(masked_y_true, masked_y_pred)
        roc = roc_auc_score(masked_y_true, masked_y_pred)
        return prc, roc
    else:
        return None, None

def compute_reg_metric(targets, preds):
    masked_targets = (targets[targets != None]).astype(np.float32)
    masked_preds = (preds[targets != None]).astype(np.float32)
    mae = mean_absolute_error(masked_targets, masked_preds)
    rmse = np.sqrt(mean_squared_error(masked_targets, masked_preds))
    return mae, rmse