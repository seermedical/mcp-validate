import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from typing import Dict


def get_responses_counts(question: str, input_dict: Dict) -> Dict:
    """Returns the count of patients that have a responded
    to a selected question.

    Args
        question: Question to select.
        input_dict: Input data of patient's responses.

    Returns
        int: Total number of responses to the selected question.
    """
    return len([patient[question] for patient in input_dict if patient[""]])


def get_diagnosis_counts(output_array: np.ndarray, col_idx: int) -> int:
    """Returns count of 1s indicating a positive diagnosis.

    Args:
        array_slice: An nx1 array, i.e. column of array,
            to count over.

    Returns:
        int: Total of counts."""

    return np.count_nonzero(output_array[:, col_idx] == 1)


def get_accuracy(
    predicted_array: np.ndarray,
    true_array: np.ndarray,
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


def get_metrics(input_dict: Dict, predicted_array: np.ndarray, true_array: np.ndarray):
    """Computes and returns a high-level statistical summary and
    performance metrics.

    Args:
        input_dict: Input data of patient's responses.
        predicted_array: Predicted output of patient's diagnoses.
        true_array: True output of patient's diagnoses.

    Returns:
        dict: A dictionary of summary and performance statistics.
    """

    pred_output = predicted_array[:, 1:3]
    true_output = true_array[:, 1:3]

    return {
        "Name": "Evaluation 1",
        "Description": "Metrics for Epilepsy vs Non-Epilepsy classes.",
        "Summary": {
            "total": {"predicted": pred_output.size, "true": true_output.size},
            "total_classes": {
                "predicted": pred_output.shape[1],
                "true": true_output.shape[1],
            },
        },
        "Counts": {
            "responses": {
                "What other things do you experience right before or at the beginning of a seizure?": get_responses_counts(
                    "What other things do you experience right before or at the beginning of a seizure?",
                    input_dict,
                ),
                "Please describe what you feel right before or at the beginning of a seizure.": 0,
                "Please specify other warning.": 0,
                "Which warnings do you get before you have a seizure?": 0,
            },
            "diagnoses": {
                "predicted": {
                    "indeterminate": get_diagnosis_counts(pred_output, 0),
                    "non_epilepsy": get_diagnosis_counts(pred_output, 1),
                    "epilepsy": get_diagnosis_counts(pred_output, 2),
                },
                "true": {
                    "indeterminate": get_diagnosis_counts(true_output, 0),
                    "non_epilepsy": get_diagnosis_counts(true_output, 1),
                    "epilepsy": get_diagnosis_counts(true_output, 2),
                },
            },
        },
        "Performance": {
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
        },
    }
