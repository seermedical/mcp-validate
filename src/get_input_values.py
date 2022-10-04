import numpy as np
from typing import Mapping, Sequence


def split_values(input_dict: dict):
    return ' '.join(list(input_dict.values())).split()


def get_flag_value(input_dict: Mapping[str, str],
                   list_of_keywords: Sequence[str]):

    # Return NaN if no answers provided to key questions
    input_values = input_dict.keys()
    if not any(input_values):
        return np.nan

    list_of_key_words = [
        'dizzy', 'dizzyness', 'dissy', 'pale', 'faint', 'light'
    ]

    split_words = split_values(input_values)

    return any(
        [keyword for keyword in list_of_keywords if keyword in split_words])


# Notes: Will need to be more sophisticated than this, e.g. requires
# an AND for questions involving duration