import numpy as np
import pytest
import spacy

from src.generate_inputs import matches_criteria, KEYWORDS_DICT, transform_input


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


@pytest.fixture
def patient_5_dict():
    return mock_input_dict_template(
        response_0="I get dizzy.",
        response_1="Usually when I go to the toilet.",
        response_6="I faint.",
    )


@pytest.fixture
def patient_6_dict():
    return mock_input_dict_template(
        response_1="Usually when I go to the toilet.", response_6="Blacking out."
    )


@pytest.fixture
def patient_7_dict():
    return mock_input_dict_template(
        response_0="I get a headache and somtimes a bit dizzy!",
        response_1="Usually when I go to the toilet.",
        response_6="Blacking out.",
    )


class TestGetInputValues:
    """Tests function get_flag_value() to check keywords in free text answer,
    and return True, False, or None based on a positive, negative, or
    no answer respectively.
    """

    nlp = spacy.load("en_core_web_sm")

    @pytest.mark.parametrize(
        "mock_response_dict, expected_result",
        [
            (
                pytest.lazy_fixture("patient_1_dict"),
                False,
            ),  # Tests that no keywords exist in answer
            (
                pytest.lazy_fixture("patient_2_dict"),
                True,
            ),  # Tests that keywords exist in answer
            (
                pytest.lazy_fixture("patient_3_dict"),
                True,
            ),  # Tests that split keywords exist in answer
            (pytest.lazy_fixture("patient_4_dict"), None),  # Tests no answer
        ],
    )
    def test_matches_one_criteria(self, mock_response_dict, expected_result):
        """Tests the function set up to match one keyword in patient's answers,
        as required by flag 1 criteria.
        """

        result = matches_criteria(
            nlp=self.nlp,
            response_dict=mock_response_dict["patient_id"],
            keywords_dict=KEYWORDS_DICT[0],
        )

        assert result == expected_result

    @pytest.mark.parametrize(
        "mock_response_dict, expected_result",
        [
            (
                pytest.lazy_fixture("patient_5_dict"),
                True,
            ),  # Tests that both keywords exist in answer
            (
                pytest.lazy_fixture("patient_6_dict"),
                True,
            ),  # Tests that both keywords exist in answer
        ],
    )
    def test_matches_two_criteria(self, mock_response_dict, expected_result):
        """Tests the function set up to match two keywords in patient's answers,
        as required by flag 2 criteria.
        """

        result = matches_criteria(
            nlp=self.nlp,
            response_dict=mock_response_dict["patient_id"],
            keywords_dict=KEYWORDS_DICT[1],
        )

        assert result == expected_result


class TestTransformInput:
    """Test function where a dictionary of patients' questions
    and answers are transformed to a one-hot
    encoded array.
    """

    @pytest.mark.parametrize(
        "mock_response_dict, expected_output",
        [
            (
                pytest.lazy_fixture("patient_7_dict"),
                [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ),  # Tests for flags 1, 2, and 5
        ],
    )
    def test_transform_input(self, mock_response_dict, expected_output):

        result = transform_input(input_dict=mock_response_dict)
        expected = np.array([expected_output]).astype(float)

        np.testing.assert_array_equal(result, expected)
