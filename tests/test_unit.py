import numpy as np
import pytest

from src.generate_inputs import (
    InputFilter,
    transform_input,
    BEFORE_EVENT_QUESTIONS,
    FLAG_6_KEYWORDS_BEFORE,
)


def mock_input_dict_template(
    response_0="",
    response_1="",
    response_2="",
    response_3="",
    response_4="",
    response_5="",
    response_6="",
    response_7="",
):
    return {
        "patient_id": {
            "What other things do you experience right before or at the beginning of a seizure?": response_0,
            "Please describe what you feel right before or at the beginning of a seizure.": response_1,
            "Please specify other warning.": response_2,
            "Please specify other injuries.": response_3,
            "What injuries have you experienced during a seizures": response_4,
            "Please specify other symptoms.": response_5,
            "Describe what happens during your seizures.": response_6,
            "How long do your seizures last?": response_7,
        }
    }


@pytest.fixture
def patient_1_dict():
    return mock_input_dict_template(
        response_1="I get a headache.", response_7="a few seconds"
    )


@pytest.fixture
def patient_2_dict():
    return {
        "What other things do you experience right before or at the beginning of a seizure?": "",
        "Please describe what you feel right before or at the beginning of a seizure.": "I feel pain, then I fall.",
        "Please specify other warning.": "",
        "Please specify other injuries.": "",
        "What injuries have you experienced during a seizure?s": "",
        "Please specify other symptoms.": "",
        "Describe what happens during your seizures.": "",
        "How long do your seizures last?": "a few seconds",
    }


@pytest.fixture
def patient_3_dict():
    return {
        "What other things do you experience right before or at the beginning of a seizure?": "",
        "Please describe what you feel right before or at the beginning of a seizure.": "",
        "Please specify other warning.": "",
        "Please specify other injuries.": "",
        "What injuries have you experienced during a seizure?s": "",
        "Please specify other symptoms.": "",
        "Describe what happens during your seizures.": "",
        "How long do your seizures last?": "a few seconds",
    }


class TestGetInputValues:
    """Tests function get_flag_value() to check keywords in free text answer,
    and return True, False, or None based on a positive, negative, or
    no answer respectively.
    """

    @pytest.mark.parametrize(
        "mock_patient_dict, expected_boolean",
        [
            (pytest.lazy_fixture("patient_1_dict"), False),
            (pytest.lazy_fixture("patient_2_dict"), True),
            (pytest.lazy_fixture("patient_3_dict"), None),
        ],
    )
    def test_get_input_values(self, mock_patient_dict, expected_boolean):
        input_filter = InputFilter(patient_dict=mock_patient_dict)
        result = input_filter.get_flag_value(
            list_of_keys=BEFORE_EVENT_QUESTIONS, list_of_keywords=FLAG_6_KEYWORDS_BEFORE
        )

        assert result == expected_boolean


class TestTransformInput:
    """Test for transform_input() function where a dictionary
    of patients' questions and answers are transformed to a one-hot
    encoded array.
    """

    def test_transform_input(self, mock_input_dict):

        result = transform_input(input_dict=mock_input_dict)
        expected = np.array([[0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]]).astype(float)

        np.testing.assert_array_equal(result, expected)
