"""
Model adapted from Beniczky S, et. al. A web-based algorithm to rapidly classify seizures for
the purpose of drug selection. Epilepsia. 2021 Oct;62(10):2474-2484. doi: 10.1111/epi.17039.
Epub 2021 Aug 22. PMID: 34420206.

Online tool: epipick.org
"""

import numpy as np
from typing import Mapping

from get_input_values import get_flag_value

FLAG_1_KEYS = ['']
FLAG_2_KEYS = ['']
FLAG_3_KEYS = ['']
FLAG_4_KEYS = ['']
FLAG_5_KEYS = ['']
FLAG_6_KEYS = ['']

FLAG_1_KEYWORDS = ['']
FLAG_2_KEYWORDS = ['']
FLAG_3_KEYWORDS = ['']
FLAG_4_KEYWORDS = ['']
FLAG_5_KEYWORDS = ['']
FLAG_6_KEYWORDS = ['']


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


def transform_input(input_dict: Mapping[str, Mapping[str, str]]):
    """Takes dictionary of patient's responses to survey questions
    and transforms to One Hot Encoded matrix."

    Args:
        input_dict (dict): Dictionary where keys represent a patient,
        enclosing another dictionary of key (question), value (answer) pairs.
        N.b. If no value (answer), to a key (question), an empty string is expected.
        Example:
            {
                'patient_1': {
                    'How long do your seizures last?': 'a few seconds',
                    'Describe what happens during your seizures.': '',
                    ...
                }
            }
      
    Returns:
        np.ndarray: Input array where rows represent each patient, and columns
        represent each input (i.e. question). Inputs are as follows:
            input_1 - Did skin turn pale before event?
            input_2 - Before event included urination or defacation, AND event included loss of
                        consciousness
            input_3 - Event duration was < 10 sec, AND event included loss of awareness and
                        fall / slump
            input_4 - Event duration was > 10 min, AND event included eyes closed
            input_5 - Before event included severe headache
            input 6 - Before event included standing up OR sit up OR posture change OR coughing
                        OR pain, AND event included falling
            input_7 - Has grey matter lesion (via imaging)
            input_8 - Event included lip smacking OR chewing
            input_9 - Events are nocturnal-only
            input_10 - Onset >= 21 y.o.
            input_11 - Event duration < 20 sec, AND event included staring OR blank OR unresponsive
                        OR unaware, AND after event did not include confusion
            input_12 - Before event excluded resting NOR sleeping AND event included jerks
            input_13 - Before event included waking w/in 1 hr OR jerking AND event included
                        convulsions on both sides, stiffening, jerks

            Elements are represented as NaN = no data, 0 = 'no', or 1 = 'yes'.
            Example:
                # +--------+--------+--------+--------+--------+--------+--------+------+----------------+----------+---------+-------+--------------+
                # | flag_1 | flag_2 | flag_3 | flag_4 | flag_5 | flag_6 | lesion | lips | night_seizures | onset_21 | staring | jerks | tonic_clonic |
                # +--------+--------+--------+--------+--------+--------+--------+------+----------------+----------+---------+-------+--------------+
                # | NaN    | NaN    | NaN    | NaN    | NaN    | NaN    | 1      | 0    | 0              | 0        | 0       | 0     | 0            |
                # +--------+--------+--------+--------+--------+--------+--------+------+----------------+----------+---------+-------+--------------+
                # | 1      | 1      | 1      | 0      | 0      | 0      | 1      | 0    | 0              | 0        | 0       | 0     | 0            |
                # +--------+--------+--------+--------+--------+--------+--------+------+----------------+----------+---------+-------+--------------+
                # | 1      | 1      | 1      | 0      | 0      | 0      | 0      | 0    | 0              | 0        | 1       | 1     | 0            |
                # +--------+--------+--------+--------+--------+--------+--------+------+----------------+----------+---------+-------+--------------+
    """
    patients = input_dict.keys()

    # Init (transformed) One Hot Encoded input array
    input_array = np.zeros(len(patients), 13)

    for idx, patient in enumerate(patients):
        patient_dict = input_dict[patient]

        input_array[idx, 0] = get_flag_value(
            input_dict={key: patient_dict[key]
                        for key in FLAG_1_KEYS},
            list_of_keywords=FLAG_1_KEYWORDS)
        input_array[idx, 1] = get_flag_value(
            {key: patient_dict[key]
             for key in FLAG_2_KEYS},
            list_of_keywords=FLAG_2_KEYWORDS)
        input_array[idx, 2] = get_flag_value(
            {key: patient_dict[key]
             for key in FLAG_3_KEYS},
            list_of_keywords=FLAG_3_KEYWORDS)
        input_array[idx, 3] = get_flag_value(
            {key: patient_dict[key]
             for key in FLAG_4_KEYS},
            list_of_keywords=FLAG_4_KEYWORDS)
        input_array[idx, 4] = get_flag_value(
            {key: patient_dict[key]
             for key in FLAG_5_KEYS},
            list_of_keywords=FLAG_5_KEYWORDS)
        input_array[idx, 5] = get_flag_value(
            {key: patient_dict[key]
             for key in FLAG_6_KEYS},
            list_of_keywords=FLAG_6_KEYWORDS)

    return input_array.astype(int)


def generate_output(input_array: np.ndarray):
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
        # +--------------+----------+-------+-------------+---------+-----------+------+---------+
        # | non_epilepsy | epilepsy | focal | generalized | absence | myoclonic | GTCS | unknown |
        # +--------------+----------+-------+-------------+---------+-----------+------+---------+
        # | 0            | 0        | 0     | 0           | 0       | 0         | 0    | 0       |
        # +--------------+----------+-------+-------------+---------+-----------+------+---------+
        # | 0            | 1        | 0     | 0           | 0       | 0         | 0    | 0       |
        # +--------------+----------+-------+-------------+---------+-----------+------+---------+
        # | 0            | 0        | 0     | 1           | 1       | 1         | 0    | 0       |
        # +--------------+----------+-------+-------------+---------+-----------+------+---------+
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


def set_diagnosis(output_array):
    # list of possible billing codes for each patient
    return


if __name__ == "__main__":

    # Create mock input data
    mock_input_data = np.zeros((5, 13))

    # Run model
    mock_output_data = generate_output(mock_input_data)
