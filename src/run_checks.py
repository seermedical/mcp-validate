from typing import Any, Dict
import pandas as pd

from src.generate_inputs import QUESTIONS_DICT


def run_checks(input_data: Dict, input_billing_codes: Dict) -> Any:

    # Check 1: n of patients in both files is equal
    if len(input_data) == len(input_billing_codes):
        print("Check 1/3: Success.")
    else:
        raise Exception(
            "Check 1/3: Failed. The length of input responses and input billing codes do not match."
        )

    # Check 2: n of unique questions is 6
    input_questions = []
    expected_questions = (
        QUESTIONS_DICT["before"] + QUESTIONS_DICT["during"] + QUESTIONS_DICT["duration"]
    )

    for patient in input_data:
        input_questions.extend(list(input_data[patient].keys()))
    if all(
        [
            question
            for question in expected_questions
            if question in set(input_questions)
        ]
    ):
        print("Check 2/3: Success.")
    else:
        raise Exception(
            "Check 2/3: Failed. The input and expected questions did not match."
        )

    # Check 3: Check all patients have all expected questions, even if no response.
    question_counts = pd.Series(input_questions).value_counts()
    for question_count in set(question_counts.values):
        if question_count != len(input_data):
            raise Exception(
                "Check 3/3: Failed. The n of input questions per patient did not match the total n of patients."
            )
    print("Check 3/3: Success.")
