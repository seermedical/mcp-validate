"""
Script of functions to generate input matrix.
"""
from typing import Mapping, Sequence
import numpy as np

FOCAL_EPILEPSY_BILLING_CODES = ["G40.0", "G40.1", "G40.2", "G40.5", "G41.2"]
GENERALISED_EPILEPSY_BILLING_CODES = [
    "G40.7",
    "G41.1",
    "G40.3",
    "G40.6",
    "G41.0",
    "G40.4",
]
UNKNOWN_EPILEPSY_BILLING_CODES = ["G40.8", "G40.9", "G41.8", "G41.9"]


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
            FOCAL_EPILEPSY_BILLING_CODES,
            GENERALISED_EPILEPSY_BILLING_CODES,
            UNKNOWN_EPILEPSY_BILLING_CODES,
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

        input_block_1 = row[0:6]
        input_block_2 = row[6:9]
        input_block_3 = row[10:13]

        # Block 1
        # Epilepsy vs Non-Epilepsy
        if has_undefined_values(input_block_1, threshold=3):
            continue

        if has_positive_values(input_block_1):
            predicted_output[idx, 0] = 1  # non-epilepsy
        else:
            predicted_output[idx, 1] = 1  # epilepsy

        # Block 2
        # Focal vs Generalised
        if has_undefined_values(input_block_2, threshold=2):
            continue

        if has_positive_values(row[9]) or has_positive_values(input_block_2):
            predicted_output[idx, 2] = 1  # focal diagnosis
            continue

        # Block 3
        # Generalised vs Unknown Onset
        # TODO: Check with Mark that we can't differentiate focal from general ?
        if has_undefined_values(input_block_3, threshold=2):
            continue

        if not has_positive_values(input_block_3):
            predicted_output[idx, 7] = 1  # unknown onset
            continue

        if has_positive_values(row[10]):
            predicted_output[idx, 4] = 1  # absence

        if has_positive_values(row[11]):
            predicted_output[idx, 5] = 1  # myoclonic

        if has_positive_values(row[12]):
            predicted_output[idx, 6] = 1  # gtcs

        if has_positive_values(predicted_output[4:7]):
            predicted_output[idx, 3] = 1  # generalised

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
