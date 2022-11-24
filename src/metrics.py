import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def get_accuracy(
    predicted_array: np.ndarray,
    true_array: np.ndarry,
    balanced: bool = False,
    normalize: bool = False,
) -> float:
    """Returns accuracy of predicted output as a percentage.

    Args:
        predicted_array: Predicted output of diagnoses.
        true_array: True output of diagnoses.

    Returns:
        score: Percentage of diagnoses that were correctly classified.
    """
    if balanced:
        score = balanced_accuracy_score(y_true=true_array, y_pred=predicted_array)
    else:
        score = accuracy_score(
            y_true=true_array, y_pred=predicted_array, normalize=normalize
        )
    return score


def get_metrics(predicted_array: np.ndarray, true_array: np.ndarry):
    """Computes and returns a high-level statistical summary and
    performance metrics.

    Args:
        predicted_array: Predicted output of diagnoses.
        true_array: True output of diagnoses.
    """

    outputs = ["predicted", "true"]

    # Evaluation 1: Run metrics for epilepsy vs non-epilepsy only
    pred_output = predicted_array[:, 1:3]
    true_output = true_array[:, 1:3]

    metrics = {
        "Name": "Evaluation 1",
        "Description": "Metrics for Epilepsy vs Non-Epilepsy classes.",
        "total": {"predicted": pred_output.size, "true": true_output.size},
        "total_classes": {
            "predicted": pred_output.shape[1],
            "true": true_output.shape[1],
        },
        "accuracy": {
            "total": get_accuracy(pred_output, true_output),
            "percentage": get_accuracy(pred_output, true_output, normalize=True),
        },
        "accuracy_balanced": {
            "total": get_accuracy(pred_output, true_output, balanced=True),
            "percentage": get_accuracy(
                pred_output, true_output, balanced=True, normalize=True
            ),
        },
    }

    return metrics
