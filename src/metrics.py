import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from typing import Dict, Sequence


def get_inputs_by_diagnosis(
    input_array: np.ndarray, true_array: np.ndarray
) -> Sequence:
    """Returns a tuple of lists of patient IDs for each
    diagnostic cohort.

    Args:
        input_array: Input data where rows represent each patient, and columns
            represent each input.
        true_array: True output of diagnoses.

    Returns:
        Sequence: Set of filtered input arrays, where each
            array represents a cohort of patients in the
            order of "Indeterminate", "Non Epilepsy" and
            "Epilepsy".
    """
    # TODO: add for focal, generalised, and unknown

    return (
        input_array[np.where(true_array[:, 0] == 1)],
        input_array[np.where(true_array[:, 1] == 1)],
        input_array[np.where(true_array[:, 0] == 2)],
    )


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


def get_input_counts(input_arrays: np.ndarray, input_idx: int) -> Dict:
    """For each diagnostic cohort, counts the number of 1s, 0s, and NaN
    inputs used for the model.

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


def get_metrics(
    input_dict: Dict,
    input_array: np.ndarray,
    predicted_array: np.ndarray,
    true_array: np.ndarray,
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

    pred_output = predicted_array[:, 1:3]
    true_output = true_array[:, 1:3]

    metrics = {
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
            "responses": {},
            "inputs": {},
            "diagnoses": {
                "predicted": {
                    "indeterminate": get_counts(pred_output, 0, 1),
                    "non_epilepsy": get_counts(pred_output, 1, 1),
                    "epilepsy": get_counts(pred_output, 2, 1),
                },
                "true": {
                    "indeterminate": get_counts(true_output, 0, 1),
                    "non_epilepsy": get_counts(true_output, 1, 1),
                    "epilepsy": get_counts(true_output, 2, 1),
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

    inputs_array_by_diagnosis = get_inputs_by_diagnosis(input_array, true_array)

    for input_idx in range(len(metrics["Counts"]["inputs"])):
        metrics["Counts"]["inputs"][input_idx] = get_input_counts(
            inputs_array_by_diagnosis, input_idx
        )

    return metrics
