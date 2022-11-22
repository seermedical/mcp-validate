"""
Script of functions to intake a dictionary of input data and transform
to a One Hot Encoded matrix of input data.
"""

from typing import List, Mapping, Union

import numpy as np
import re

QUESTIONS_DICT = {
    "before": [
        "What other things do you experience right before or at the beginning of a seizure?",
        "Please describe what you feel right before or at the beginning of a seizure.",
        "Please specify other warning.",
        "Which warnings do you get before you have a seizure?",
    ],
    "during": [
        "Please specify other symptoms.",
        "Describe what happens during your seizures.",
    ],
    "duration": ["How long do your seizures last?"],
}


KEYWORDS_DICT = {
    0: {
        "before": [
            "pale",
            "white",
            "vertigo",
            r"di(zz|ss)y",
            r"light[ -]?head(ed)?",
        ],
    },
    1: {
        "before": ["toilet", "restroom"],
        "during": ["conscious", "fall", "aware", "faint", r"black(ed|ing)?[ -]?out"],
    },
    2: {"during": ["collapse", "droop", "slump"]},
    3: {
        "during": ["eye", "close", "shut"],
        "duration": [
            "7 - 15 minutes",
            "more than 15 minutes",
        ],
    },
    4: {"before": ["migraine", r"head ?ache"]},
    5: {"before": ["pain", "cough", "stand"], "during": ["fell", "fall"]},
}


def search_keywords(
    input_text: str,
    patterns: List[Union[str, tuple]],
) -> Union[bool, None]:
    """Determines if any keywords exist in a list of words.

    Takes a list of words derived from a patient's response/s and searches
    for specific keywords. In some instances, multiple keywords are
    required to determine a match, e.g. "black" and "out" in the instance
    of "black out".

    Args:
        input_text: Concatenated string of words from patient's response/s
        patterns: List of keywords to match.

    Returns:
        bool: Returns True if any keywords in list of words, else False.
    """

    while True:
        for pattern in patterns:
            # Search for Regex pattern
            pattern_exists = re.search(re.compile(pattern), input_text)
            if pattern_exists:
                return True
        return False


def matches_criteria(
    response_dict: Mapping[str, str],
    keywords_dict: Mapping[str, List[Union[str, tuple]]],
) -> bool:
    """Determines if patient reponses fill given criteria for a particular input.

    Takes dict of a patient's responses and returns True if a patient
    matches a given set of criteria, and False if the patient does not match a
    given set of criteria.

    Args:
        response_dict: A patient's responses to a set of questions. Key-value pairs
            represent questions and responses.
            Example: {
                "Question 1": "Response 1",
                "Question 2": "Response 2",
                ...
            }
        keywords_dict: A set of criteria required for a given input. Key-value pairs
            represent an input's criteria category and criteria keywords. The criteria
            category, e.g. "before" is used to determine which questions from the list of
            available questions is relevant.
            Example: {
                "before": ["toilet", "restroom"],
                "during": ["conscious", "fall", "aware", "faint", "blackout", ("black", "out")]
                }

    Returns:
        bool: Returns True if all criteria is matched, elif False if no criteria
            is matched, else None if no relevant questions are answered.
    """

    matched_criteria = []

    # Filter response_dict (patient's responses) to relevant questions only
    for criteria, patterns in keywords_dict.items():
        relevant_questions = QUESTIONS_DICT[criteria]
        relevant_responses = {
            k: v for k, v in response_dict.items() if k in relevant_questions
        }

        if not any(relevant_responses.values()):
            matched_criteria.append(None)
            continue

        # Add all responses to a single string
        input_text = " ".join(relevant_responses.values())
        # From the single string, remove punctuation and cast to lower case
        input_text = re.sub(r"[^\w\s]", "", input_text.lower())

        # Search for keywords in patient's responses
        matched_criteria.append(search_keywords(input_text, patterns))

    # Return None if no responses to relevant questions
    if matched_criteria.count(None) == len(matched_criteria):
        return None

    return all(matched_criteria)


def transform_input(input_dict: Mapping[str, Mapping[str, str]]) -> np.ndarray:
    """Takes dictionary of patient responses and generates a
    One Hot Encoded matrix."

        Args:
            input_dict: Dictionary where keys represent a patient,
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
            input_array: Array of input data where rows represent each patient, and columns
                represent each input (i.e. question).
                Inputs are as follows:
                Input 1 - Did skin turn pale before event?
                Input 2 - Before event included urination or defacation, AND event included loss of
                    consciousness.
                Input 3 - Event duration was < 10 sec, AND event included loss of awareness and
                    fall / slump
                Input 4 - Event duration was > 10 min, AND event included eyes closed
                Input 5 - Before event included severe headache
                Input 6 - Before event included standing up OR sit up OR posture change OR coughing
                    OR pain, AND event included falling

                Elements are represented as NaN = no data, 0 = 'no', or 1 = 'yes'.
                Example:
                    # +--------+--------+--------+--------+--------+--------+
                    # | flag_1 | flag_2 | flag_3 | flag_4 | flag_5 | flag_6 |
                    # +--------+--------+--------+--------+--------+--------+
                    # | NaN    | NaN    | NaN    | NaN    | NaN    | NaN    |
                    # +--------+--------+--------+--------+--------+--------+
                    # | 1      | 1      | 1      | NaN    | 0      | 0      |
                    # +--------+--------+--------+--------+--------+--------+
                    # | 1      | 1      | 0      | 0      | 1      | 1      |
                    # +--------+--------+--------+--------+--------+--------+
    """

    # Init (transformed) One Hot Encoded input array
    input_array = np.zeros([len(input_dict), 13])

    for row_idx, patient_dict in enumerate(input_dict.values()):

        # Set row to np.nan if no responses
        if not any(patient_dict.values()):
            input_array[row_idx, :] = np.nan

        for col_idx, keywords_dict in KEYWORDS_DICT.items():

            input_array[row_idx, col_idx] = matches_criteria(
                patient_dict, keywords_dict
            )

    return input_array.astype(float)
