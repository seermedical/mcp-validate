"""
Model adapted from Beniczky S, et. al. A web-based algorithm to rapidly classify seizures for
the purpose of drug selection. Epilepsia. 2021 Oct;62(10):2474-2484. doi: 10.1111/epi.17039.
Epub 2021 Aug 22. PMID: 34420206.

Online tool: epipick.org

Authors: Dominique Eden & Pip Karoly
"""

from typing import Mapping, Sequence

from src.generate_inputs import transform_input
from src.generate_outputs import get_predicted_output, get_true_output


def run(
    input_data: Mapping[str, Mapping[str, str]],
    input_billing_codes: Mapping[str, Sequence[str]],
) -> None:

    # Get input array
    input_array = transform_input(input_data)

    # Run model
    predicted_output = get_predicted_output(input_array)
    true_output = get_true_output(input_billing_codes)

    return


if __name__ == "__main__":

    # Create mock input data
    mock_input = {"patient_1": {"question_1": "answer_1"}}
    mock_diagnosis = {"patient_1": ["billing_code_1", "billing_code_2"]}

    run(mock_input, mock_diagnosis)
