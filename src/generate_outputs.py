"""
Script of functions to generate input matrix.
"""

import numpy as np
from typing import Mapping, Sequence

FOCAL_EPILEPSY_BILLING_CODES = ['G40.0', 'G40.1', 'G40.2', 'G40.5', 'G41.2']
GENERALISED_EPILEPSY_BILLING_CODES = [
    'G40.7', 'G41.1', 'G40.3', 'G40.6', 'G41.0', 'G40.4'
]
UNKNOWN_EPILEPSY_BILLING_CODES = ['G40.8', 'G40.9', 'G41.8', 'G41.9']
EPILEPSY_BILLING_CODES = FOCAL_EPILEPSY_BILLING_CODES + GENERALISED_EPILEPSY_BILLING_CODES + UNKNOWN_EPILEPSY_BILLING_CODES


def set_diagnosis(list_of_billing_codes: Sequence[str]):
    """_summary_

    Args:
        list_of_billing_codes (Sequence[str]): _description_
    """

    # Init row for np.array
    diagnosis_array = np.zeros([1, 5])

    # Populate rows
    if any([
            billing_code for billing_code in EPILEPSY_BILLING_CODES
            if billing_code in list_of_billing_codes
    ]):
        diagnosis_array[0, 1] = 1
    else:
        diagnosis_array[0, 0] = 1

        return diagnosis_array

    if any([
            billing_code for billing_code in FOCAL_EPILEPSY_BILLING_CODES
            if billing_code in list_of_billing_codes
    ]):
        diagnosis_array[0, 2] = 1
    if any([
            billing_code for billing_code in GENERALISED_EPILEPSY_BILLING_CODES
            if billing_code in list_of_billing_codes
    ]):
        diagnosis_array[0, 3] = 1
    if any([
            billing_code for billing_code in UNKNOWN_EPILEPSY_BILLING_CODES
            if billing_code in list_of_billing_codes
    ]):
        diagnosis_array[0, 4] = 1

    return diagnosis_array


def has_undefined_values(input_array: np.ndarray, threshold: int = 3):
    """Checks if an array has enough valid data given a threshold.

    Returns:
        bool: Returns True if n of NaN elements exceeds threshold, and False if
                n of NaN elements does not exceed threshold.
    """
    return np.count_nonzero(np.isnan(input_array)) > threshold


def has_positive_values(input_array: np.ndarray, threshold: int = 0):
    """Checks if an array has enough non-zero data given a threshold.

    Returns:
        bool: Returns True if n of non-zero elements exceeds threshold, and False if
                n of non-zero elements does not exceed threshold.
    """
    return np.count_nonzero(input_array) > threshold


def get_predicted_output(input_array: np.ndarray):
    """Predicts diagnosis of each patient based on a set of inputs.

    Args:
        np.ndarray: Input array where rows represent each patient, and columns
        represent each input (i.e. question). See to transform_input().

    Returns:
        np.ndarray: Output array where rows represent patients and columns represent
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
    output_array = np.zeros((n_rows, 8))

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
            output_array[idx, 0] = 1  # non-epilepsy
        else:
            output_array[idx, 1] = 1  # epilepsy

        # Block 2
        # Focal vs Generalised
        if has_undefined_values(input_block_2, threshold=2):
            continue

        if has_positive_values(row[9]) or has_positive_values(input_block_2):
            output_array[idx, 2] = 1  # focal diagnosis
            continue

        # Block 3
        # Generalised vs Unknown Onset
        # TODO: Check with Mark that we can't differentiate focal from general ?
        if has_undefined_values(input_block_3, threshold=2):
            continue

        if not has_positive_values(input_block_3):
            output_array[idx, 7] = 1  # unknown onset
            continue

        if has_positive_values(row[10]):
            output_array[idx, 4] = 1  # absence

        if has_positive_values(row[11]):
            output_array[idx, 5] = 1  # myoclonic

        if has_positive_values(row[12]):
            output_array[idx, 6] = 1  # gtcs

        if has_positive_values(output_array[4:7]):
            output_array[idx, 3] = 1  # generalised

    return output_array


def get_true_output(input_billing_codes: Mapping[str, Sequence[str]]):
    """Defines diagnosis for each patient based on a set of billing codes.

    Args:
        dict: Dictionary of patient's billing codes.

    Returns:
        np.ndarray: Output array where rows represent patients and columns represent
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
    output_array = np.zeros([input_billing_codes.__len__(), 5])

    for idx, patient in enumerate(patient_keys):
        patient_values = input_billing_codes[patient].values()

        output_array[idx] = set_diagnosis(list_of_billing_codes=patient_values)

    true_outputs = output_array.astype(int)

    return true_outputs