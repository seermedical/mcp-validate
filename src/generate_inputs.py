"""
Script of functions to generate predicted and true output matrices.
"""

from typing import Mapping, Sequence

import numpy as np

QUESTION_1 = 'What other things do you experience right before or at the beginning of a seizure?'
QUESTION_2 = 'Please describe what you feel right before or at the beginning of a seizure.'
QUESTION_3 = 'Please specify other warning.'
QUESTION_4 = 'Please specify other injuries.'
QUESTION_5 = 'What injuries have you experienced during a seizure?'
QUESTION_6 = 'Please specify other symptoms.'
QUESTION_7 = 'Describe what happens during your seizures.'
QUESTION_8 = 'Please describe what you feel right before or at the beginning of a seizure.'
QUESTION_9 = 'How long do your seizures last?'

FLAG_1_KEYS = ['']
FLAG_2_KEYS = ['']
FLAG_3_KEYS = ['']
FLAG_4_KEYS = ['']
FLAG_5_KEYS = ['']
FLAG_6_KEYS = ['']

FLAG_1_KEYWORDS = ['dizzy', 'dizzyness', 'dissy', 'pale', 'faint', 'light']
FLAG_2_KEYWORDS = ['']
FLAG_3_KEYWORDS = ['']
FLAG_4_KEYWORDS = ['']
FLAG_5_KEYWORDS = ['']
FLAG_6_KEYWORDS = ['']


def split_values(input_dict: dict):
    return ' '.join(list(input_dict.values())).split()


def get_flag_value(input_dict: Mapping[str, str],
                   list_of_keywords: Sequence[str]):

    # Return NaN if no answers provided to key questions
    input_values = input_dict.keys()
    if not any(input_values):
        return np.nan

    split_words = split_values(input_values)

    return any(
        [keyword for keyword in list_of_keywords if keyword in split_words])


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
=
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
    input_array = np.zeros(len(input_dict), 13)

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


# Notes: Will need to be more sophisticated than this, e.g. requires
# an AND for questions involving duration