"""
Script of functions to generate input matrix.
"""
from typing import Mapping, Sequence
import numpy as np

BILLING_CODES = {
    "focal": ["G40.0", "G40.1", "G40.2", "G40.5", "G41.2"],
    "generalised": [
        "G40.7",
        "G41.1",
        "G40.3",
        "G40.6",
        "G41.0",
        "G40.4",
    ],
    "unknown": ["G40.8", "G40.9", "G41.8", "G41.9"],
}


def set_diagnosis(list_of_billing_codes: Sequence[str]) -> np.ndarray:
    """Uses a list of ICD-10 billing codes to generate a row per patient
    for diagnostic array (see get_predicted_output).

    Args:
        list_of_billing_codes: List of ICD-10 billing codes
            for a patient.

    Returns:
        output_row: Row of 1s and 0s representing True or False
            diagnostic values.
    """

    # Init row
    output_row = np.zeros([1, 5])

    # Populate rows
    for i, billing_code_category in enumerate(
        [
            BILLING_CODES["focal"],
            BILLING_CODES["generalised"],
            BILLING_CODES["unk"],
        ]
    ):
        if any(
            billing_code in billing_code_category
            for billing_code in list_of_billing_codes
        ):
            output_row[0, i + 2] = 1

    if output_row.sum() == 0:
        output_row[0, 0] = 1
    else:
        output_row[0, 1] = 1

    return output_row


def has_undefined_values(input_array: np.ndarray, threshold: int = 3) -> bool:
    """Checks if an array has enough valid data given a threshold.

    Returns:
        bool: Returns True if n of NaN elements exceeds threshold, and False if
                n of NaN elements does not exceed threshold.
    """
    return np.count_nonzero(np.isnan(input_array)) > threshold


def has_positive_values(input_array: np.ndarray, threshold: int = 0) -> bool:
    """Checks if an array has enough non-zero data given a threshold.

    Returns:
        bool: Returns True if n of non-zero elements exceeds threshold, and False if
                n of non-zero elements does not exceed threshold.
    """
    return np.count_nonzero(input_array) > threshold


def get_predicted_output(input_array: np.ndarray) -> np.ndarray:
    """Predicts diagnosis of each patient based on a set of inputs.

    Args:
        input_array: Input array where rows represent each patient, and columns
            represent each input (i.e. question). See to transform_input().

    Returns:
        predicted_output: Output array where rows represent patients and columns represent
            predicted diagnosis (i.e. output classes). Outputs are as follows:
            output_1 - Non-epileptic paroxysmal event
            output_2 - Epileptic
            output_3 - Focal
            output_4 - Generalized
            output_5 - Absence
            output_6 - Myoclonic
            output_7 - GTCS (Generalized Tonic Clonic Seizures)

            Elements are represented as 0 = negative diagnosis, or 1 = positive diagnosis. N.b. A
            patient may have multiple diagnoses.
            # Example:
            # +--------------+----------+-------+-------------+---------+
            # | non_epilepsy | epilepsy | focal | generalized | unknown |
            # +--------------+----------+-------+-------------+---------+
            # | 0            | 0        | 0     | 0           | 0       |
            # +--------------+----------+-------+-------------+---------+
            # | 0            | 1        | 1     | 0           | 0       |
            # +--------------+----------+-------+-------------+---------+
            # | 0            | 1        | 0     | 1           | 0       |
            # +--------------+----------+-------+-------------+---------+
    """

    n_rows = input_array.shape[0]

    # create an output array
    predicted_output = np.zeros((n_rows, 8))

    for idx in range(n_rows):
        row = input_array[idx, :]

        # Epilepsy vs Non-Epilepsy
        if has_undefined_values(row, threshold=3):
            continue

        if has_positive_values(row):
            predicted_output[idx, 0] = 1  # non-epilepsy
        else:
            predicted_output[idx, 1] = 1  # epilepsy

    return predicted_output


def get_true_output(input_billing_codes: Mapping[str, Sequence[str]]) -> np.ndarray:
    """Defines diagnosis for each patient based on a set of billing codes.

    Args:
        input_billing_codes: Dictionary of patients' billing codes.

    Returns:
        true_output: Output array where rows represent patients and columns represent
            true diagnosis. Outputs are as follows:
            output_1 - Non-epileptic paroxysmal event
            output_2 - Epileptic
            output_3 - Focal
            output_4 - Generalized
            output_5 - Absence
            output_6 - Myoclonic
            output_7 - GTCS (Generalized Tonic Clonic Seizures)

            Elements are represented as 0 = negative diagnosis, or 1 = positive diagnosis. N.b. A
            patient may have multiple diagnoses.
            # Example:
            # +--------------+----------+-------+-------------+---------+
            # | non_epilepsy | epilepsy | focal | generalized | unknown |
            # +--------------+----------+-------+-------------+---------+
            # | 0            | 0        | 0     | 0           | 0       |
            # +--------------+----------+-------+-------------+---------+
            # | 0            | 1        | 1     | 0           | 0       |
            # +--------------+----------+-------+-------------+---------+
            # | 0            | 1        | 0     | 1           | 0       |
            # +--------------+----------+-------+-------------+---------+
    """

    patient_keys = input_billing_codes.keys()

    # Init output array for true diagnoses
    true_output = np.zeros([len(input_billing_codes), 5])

    for idx, patient in enumerate(patient_keys):
        patient_values = input_billing_codes[patient].values()

        true_output[idx] = set_diagnosis(list_of_billing_codes=patient_values)

    return true_output
