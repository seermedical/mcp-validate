import numpy as np
import pytest
from src.generate_inputs import matches_criteria, KEYWORDS_DICT, transform_input


class TestGetInputValues:
    """Tests function get_flag_value() to check keywords in free text answer,
    and return True, False, or None based on a positive, negative, or
    no answer respectively.
    """

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
            response_dict=mock_response_dict,
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
            response_dict=mock_response_dict,
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
