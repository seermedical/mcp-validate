import numpy as np
import pytest
import spacy

from src.generate_inputs import matches_criteria, KEYWORDS_DICT


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
            "Which warnings do you get before you have a seizure?": response_4,
            "Please specify other symptoms.": response_5,
            "Describe what happens during your seizures.": response_6,
            "How long do your seizures last?": response_7,
        }
    }


@pytest.fixture
def patient_1_dict():
    return mock_input_dict_template(response_1="I get a headache.")


@pytest.fixture
def patient_2_dict():
    return mock_input_dict_template(response_1="I get dizzy.")


@pytest.fixture
def patient_3_dict():
    return mock_input_dict_template(response_1="I get a bit light headed.")


@pytest.fixture
def patient_4_dict():
    return mock_input_dict_template(response_7="a few seconds")


class TestGetInputValues:
    """Tests function get_flag_value() to check keywords in free text answer,
    and return True, False, or None based on a positive, negative, or
    no answer respectively.
    """

    nlp = spacy.load("en_core_web_sm")

    @pytest.mark.parametrize(
        "mock_response_dict, expected_result",
        [
            (pytest.lazy_fixture("patient_1_dict"), False),
            (pytest.lazy_fixture("patient_2_dict"), True),
            (pytest.lazy_fixture("patient_3_dict"), True),
            (pytest.lazy_fixture("patient_4_dict"), None),
        ],
    )
    def test_matches_one_criteria(self, mock_response_dict, expected_result):

        result = matches_criteria(
            nlp=self.nlp,
            response_dict=mock_response_dict["patient_id"],
            keywords_dict=KEYWORDS_DICT[0],
        )

        assert result == expected_result


# class TestTransformInput:
#     """Test for transform_input() function where a dictionary
#     of patients' questions and answers are transformed to a one-hot
#     encoded array.
#     """

#     def test_transform_input(self, mock_input_dict):

#         result = transform_input(input_dict=mock_input_dict)
#         expected = np.array([[0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]]).astype(float)

#         np.testing.assert_array_equal(result, expected)
