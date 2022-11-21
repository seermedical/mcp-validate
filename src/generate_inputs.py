"""
Script of functions to generate predicted and true output matrices.
"""

from typing import List, Mapping, Union

import numpy as np
import spacy

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
        "before": ["pale", "white", "dizzy", "dissy", "vertigo", ("light", "head")],
    },
    1: {
        "before": ["toilet", "restroom"],
        "during": ["conscious", "fall", "aware", "faint", "blackout", ("black", "out")],
    },
    2: {"during": ["collapse", "droop", "slump"]},
    3: {
        "during": ["eye", "close", "shut"],
        "duration": [
            "7 - 15 minutes",
            "more than 15 minutes",
        ],
    },
    4: {"before": ["headache", "migraine", ("head", "ache")]},
    5: {"before": ["pain", "cough", "stand"], "during": ["fell", "fall"]},
}  # TODO: check with MCPV team that format is in str


def search_keywords(
    response_list: List[str],
    keywords_list: List[Union[str, tuple]],
) -> Union[bool, None]:
    """Determines if any keywords exist in a list of words.
    Takes a list of words derived from a patient's response/s and
    and searches for specific keywords.

    In some instances, multiple keywords are required to determine
    a match, e.g. "black" and "out" in the instance of "black out".

    Args:
        response_list: List of words derived from a patient's response/s.
        keywords_list: List of keywords to match.

    Returns:
        bool: Returns True if any keywords in list of words, else False.
    """
    matched_words = []

    for keyword in keywords_list:
        if type(keyword) == str:
            matched_words.append(any(keyword in word for word in response_list))

        else:
            matched_split_words = []
            for split_word in keyword:
                matched_split_words.append(
                    any(split_word in word for word in response_list)
                )
            matched_words.append(all(matched_split_words))

    return any(matched_words)


def matches_criteria(
    nlp: spacy.language.Language,
    response_dict: Mapping[str, str],
    keywords_dict: Mapping[str, List[Union[str, tuple]]],
) -> bool:
    """Determines if a patient's reponse/s fills given criteria for a particular input.
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
            "during": ["conscious", "fall", "aware", "faint", "blackout", ("black", "out")]}

    Returns:
        bool: Returns True if all criteria is matched, elif False if no criteria
        is matched, else None if no relevant questions are answered.
    """

    matched_criteria = []

    # Filter response_dict (patient's responses) to relevant questions only
    for criteria, keywords_list in keywords_dict.items():
        relevant_questions = QUESTIONS_DICT[criteria]
        relevant_responses = {
            k: v for (k, v) in response_dict.items() if k in relevant_questions
        }

        if not any(relevant_responses.values()):
            matched_criteria.append(None)
            continue

        # Use Spacy's NLP module to generate a list of strings from patient's responses
        input_sentences = " ".join(relevant_responses.values())
        split_words_doc = list(nlp(input_sentences))
        split_words_lst = set([token.text.lower() for token in split_words_doc])

        # Search for keywords in patient's responses
        matched_criteria.append(search_keywords(split_words_lst, keywords_list))

    # Return None if no responses to relevant questions
    if matched_criteria.count(None) == len(matched_criteria):
        return None

    return all(matched_criteria)


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

    nlp = spacy.load("en_core_web_sm")

    # Init (transformed) One Hot Encoded input array
    input_array = np.zeros(len(input_dict), 13)

    for row_idx, patient_dict in enumerate(input_dict.values()):

        # Set row to np.nan if no responses
        if not any(patient_dict.values()):
            input_array[row_idx, :] = np.nan

        for col_idx, keywords_dict in list(KEYWORDS_DICT.items()):

            input_array[row_idx, col_idx] = matches_criteria(
                nlp, patient_dict, keywords_dict
            )

    input_array = input_array.astype(int)

    return input_array
