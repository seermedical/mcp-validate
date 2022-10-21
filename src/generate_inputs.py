"""
Script of functions to generate predicted and true output matrices.
"""

from typing import List, Mapping, Optional

import numpy as np
import spacy

BEFORE_EVENT_QUESTIONS = [
    "What other things do you experience right before or at the beginning of a seizure?",
    "Please describe what you feel right before or at the beginning of a seizure.",
    "Please specify other warning."
    "Which warnings do you get before you have a seizure?",
]
DURING_EVENT_QUESTIONS = [
    "Please specify other symptoms.",
    "Describe what happens during your seizures.",
]
DURATION_QUESTIONS = ["How long do your seizures last?"]

FLAG_1_KEYWORDS = ["pale", "white", "dizzy", "dissy", "vertigo"]
FLAG_1_KEYWORDS_MULTIPLE = ["light", "head"]
FLAG_2_KEYWORDS = [""]
FLAG_3_KEYWORDS = ["collapse", "droop", "slump"]
FLAG_4_KEYWORDS_DURING = ["eye", "close", "shut"]
FLAG_4_KEYWORDS_DURATION = [
    "7 - 15 minutes",
    "more than 15 minutes",
]  # TODO: check with MCPV team that format is in str
FLAG_5_KEYWORDS = ["headache", "migraine"]
FLAG_5_KEYWORDS_MULTIPLE = ["head", "ache"]
FLAG_6_KEYWORDS_BEFORE = ["pain", "cough", "stand"]
FLAG_6_KEYWORDS_DURING = ["fell", "fall"]


class InputFilter:

    nlp = spacy.load("en_core_web_sm")

    def __init__(self, patient_dict):
        self.patient_dict = patient_dict

    def get_flag_value(
        self,
        list_of_keys: List[str],
        list_of_keywords: List[str],
        multiple_words_to_match: bool = False,
    ) -> Optional[bool]:
        # Filter the patient dict of keys (questions) and values (answers)
        input_dict = {key: self.patient_dict[key] for key in list_of_keys}

        # Return NaN if no answers provided to key questions
        input_values = input_dict.values()
        if not any(input_values):
            return None

        input_sentences = " ".join(input_values)
        split_words_doc = list(self.nlp(input_sentences))
        split_words_str = set([token.text for token in split_words_doc])

        if not multiple_words_to_match:
            return any(
                keyword for keyword in list_of_keywords if keyword in split_words_str
            )
        else:
            return all(
                keyword for keyword in list_of_keywords if keyword in split_words_str
            )


def transform_input(input_dict: Mapping[str, Mapping[str, str]]) -> np.ndarray:
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
                                consciousness.
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

    # Init (transformed) One Hot Encoded input array
    input_array = np.zeros(len(input_dict), 13)

    for idx, patient_dict in enumerate(input_dict.values()):

        filter_input = InputFilter(patient_dict=patient_dict)
        # Flag 1: Pale skin before event
        input_array[idx, 0] = any(
            [
                filter_input.get_flag_value(BEFORE_EVENT_QUESTIONS, FLAG_1_KEYWORDS),
                filter_input.get_flag_value(
                    BEFORE_EVENT_QUESTIONS,
                    FLAG_1_KEYWORDS_MULTIPLE,
                    multiple_words_to_match=True,
                ),
            ]
        )

        # Flag 2: Loss of consciousness immediately after urination
        # or defacation
        input_array[idx, 1] = filter_input.get_flag_value(
            DURING_EVENT_QUESTIONS, FLAG_2_KEYWORDS
        )

        # Flag 3: Fall or slump with loss of awareness
        # during event
        input_array[idx, 2] = filter_input.get_flag_value(
            DURING_EVENT_QUESTIONS, FLAG_3_KEYWORDS
        )

        # Flag 4: Seizure with eyes closed lasting longer than 10 minutes
        input_array[idx, 3] = all(
            [
                filter_input.get_flag_value(
                    DURING_EVENT_QUESTIONS, FLAG_4_KEYWORDS_DURING
                ),
                filter_input.get_flag_value(
                    DURATION_QUESTIONS, FLAG_4_KEYWORDS_DURATION
                ),
            ]
        )

        # Flag 5: Severe preictal headache
        input_array[idx, 4] = any(
            [
                filter_input.get_flag_value(BEFORE_EVENT_QUESTIONS, FLAG_5_KEYWORDS),
                filter_input.get_flag_value(
                    BEFORE_EVENT_QUESTIONS,
                    FLAG_5_KEYWORDS_MULTIPLE,
                    multiple_words_to_match=True,
                ),
            ]
        )

        # Flag 6: Fall after posture change, standing, coughing, or pain
        input_array[idx, 5] = all(
            [
                filter_input.get_flag_value(
                    BEFORE_EVENT_QUESTIONS, FLAG_6_KEYWORDS_BEFORE
                ),
                filter_input.get_flag_value(
                    DURING_EVENT_QUESTIONS, FLAG_6_KEYWORDS_DURING
                ),
            ]
        )

    input_array = input_array.astype(int)

    return input_array
