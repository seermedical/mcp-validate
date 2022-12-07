"""Script of functions to compute a high-level statstical summary
and performance of model."""

from typing import Dict, Sequence
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def get_inputs_by_diagnosis(
    input_array: np.ndarray, true_array: np.ndarray
) -> Sequence:
    """Returns a tuple of arrays, where each array is filtered
    to include only patients with a given diagnosis.

    Args:
        input_array: Input data where rows represent each patient, and
            columns represent each input.
        true_array: True output of diagnoses.

    Returns:
        Sequence: Set of filtered input arrays, where each array
            represents a cohort of patients in the order of
            indeterminate, non-epilepsy, and epilepsy.
    """

    return tuple(input_array[np.where(true_array[:, x] == 1)] for x in range(3))


def get_responses_counts(question: str, input_dict: Dict) -> Dict:
    """Returns the total number of patients that have a responded
    to a selected question.

    Args
        question: Question to select.
        input_dict: Input data of patient's responses.

    Returns
        int: Total number of responses to the selected question.
    """

    return len([patient[question] for patient in input_dict if patient[""]])


def get_input_counts(input_arrays: np.ndarray, input_idx: int) -> Dict:
    """Counts the number of 1s, 0s, and NaN values for each
    diagnostic class.

    Args:
        input_arrays: Set of input arrays, filtered by diagnosis.
        input_idx: The input, i.e. column, to count over.

    Returns:
        Dict: Dictionary of counts of 1s, 0s, and NaN inputs,
            per diagnostic cohort.
    """

    input_counts = {}

    for cohort_name, cohort_array in zip(
        ["indeterminate", "non_epilepsy", "epilepsy"], input_arrays
    ):
        input_counts[cohort_name] = {
            0: get_counts(cohort_array, input_idx, 0),
            1: get_counts(cohort_array, input_idx, 1),
            np.nan: get_counts(cohort_array, input_idx, np.nan),
        }
    return input_counts


def get_counts(array: np.ndarray, col_idx: int, value: float) -> int:
    """Counts the number of a given value across a column of an
    array.

    Args:
        array: Array to count over.
        col_idx: Index of column of array to count over.
        value: Value to count.


    Returns:
        int: Total of counts."""
    if np.isnan(value):
        return np.count_nonzero(np.isnan(array[:, col_idx]))

    return np.count_nonzero(array[:, col_idx] == value)


def get_accuracy(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    balanced: bool = False,
    normalize: bool = False,
) -> float:
    """Returns accuracy of predicted labels.

    Args:
        pred_labels: Predicted labels of diagnoses.
        true_labels: True labels of diagnoses.

    Returns:
        score: Score indicating the n or percentage
            correctly classified.
    """

    if balanced:
        score = balanced_accuracy_score(y_true=true_labels, y_pred=pred_labels)
    else:
        score = accuracy_score(
            y_true=true_labels, y_pred=pred_labels, normalize=normalize
        )
    return score


def get_labels(output_array: np.ndarray) -> np.ndarray:
    """Computes the labels (i.e. n of positive diagnoses)
    for each class (i.e. diagnosis).

    Args:
        output_array: The One Hot Encoded output array to
            compute labels across.

    Returns:
        np.ndarray: Labels.
    """
    return np.array(
        [
            get_counts(output_array, 0, 1),
            get_counts(output_array, 1, 1),
            get_counts(output_array, 2, 1),
        ]
    )


def get_metrics(
    input_array: np.ndarray,
    pred_output: np.ndarray,
    true_output: np.ndarray,
):
    """Computes and returns a high-level statistical summary and
    performance metrics.

    Args:
        input_dict: Input data of patient's responses.
        input_array: Input data to model, represented by flags.
        predicted_array: Predicted output of patient's diagnoses.
        true_array: True output of patient's diagnoses.

    Returns:
        dict: A dictionary of summary and performance statistics.
    """

    pred_labels = get_labels(pred_output)
    true_labels = get_labels(true_output)

    metrics = {
        "Name": "Evaluation 1",
        "Description": "Metrics for Epilepsy vs Non-Epilepsy classes.",
        "Summary": {
            "total": {"predicted": pred_output.shape[0], "true": true_output.shape[0]},
            "total_classes": {
                "predicted": pred_output.shape[1],
                "true": true_output.shape[1],
            },
        },
        "Counts": {
            "responses": {},
            "inputs": {},
            "diagnoses": {
                "predicted": {
                    "indeterminate": pred_labels[0],
                    "non_epilepsy": pred_labels[1],
                    "epilepsy": pred_labels[2],
                },
                "true": {
                    "indeterminate": true_labels[0],
                    "non_epilepsy": true_labels[1],
                    "epilepsy": true_labels[2],
                },
            },
        },
        "Performance": {
            "accuracy": {
                "total": get_accuracy(pred_labels, true_labels),
                "percentage": get_accuracy(pred_labels, true_labels, normalize=True),
            },
            "accuracy_balanced": {
                "total": get_accuracy(pred_labels, true_labels, balanced=True),
            },
        },
    }

    inputs_array_by_diagnosis = get_inputs_by_diagnosis(input_array, true_output)

    for input_idx in range(input_array.shape[1]):
        metrics["Counts"]["inputs"][input_idx] = get_input_counts(
            inputs_array_by_diagnosis, input_idx
        )

    return metrics
