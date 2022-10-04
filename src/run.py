from typing import Mapping, Sequence

from generate_inputs import transform_input
from generate_outputs import get_predicted_output, get_true_output


def run(input_data: Mapping[str, Mapping[str, str]],
        input_billing_codes: Mapping[str, Sequence[str]]):

    # Get input array
    input_array = transform_input(input_data)

    # Run model
    predicted_output = get_predicted_output(input_array)
    true_output = get_true_output(input_billing_codes)

    return


if __name__ == "__main__":

    # Create mock input data
    mock_input = {'patient_1': {'question_1': 'answer_1'}}
    mock_diagnosis = {'patient_1': ['billing_code_1', 'billing_code_2']}

    run(mock_input, mock_diagnosis)
