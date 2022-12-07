import numpy as np
import pytest

from src.generate_inputs import matches_criteria, KEYWORDS_DICT, transform_input
from src.metrics import get_inputs_by_diagnosis, get_input_counts, get_metrics


def mock_input_dict_template(
    patient_id="patient_id",
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
        patient_id: {
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


@pytest.fixture
def mock_input_dict():
    return {
        mock_input_dict_template(
            patient_id="patient_8",
            response_0="I go pale, I get a headache.",
            response_6="I droop.",
        )[
            0
        ],  # non-epilepsy
        mock_input_dict_template(
            patient_id="patient_9",
            response_0="I'm not sure",
            response_1="I'm not sure",
            response_6="I'm not sure",
            response_7="a few seconds",
        )[
            0
        ],  # epilepsy
        mock_input_dict_template(
            patient_id="patient_10", response_0="Some text", response_1="Some text"
        )[
            0
        ],  # indeterminate
    }


@pytest.fixture
def mock_input_array():
    return np.array(
        [
            [1, 0, 1, np.nan, 1, 0],  # epilepsy
            [0, 0, 0, 0, 0, 0],  # non-epilepsy
            [0, np.nan, np.nan, np.nan, 0, np.nan],  # indeterminate
        ]
    )


@pytest.fixture
def mock_true_array():
    return np.array([[0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])


@pytest.fixture
def mock_pred_array():
    return np.array([[0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])


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


class TestMetrics:
    """Tests function to compute and return metrics."""

    def test_get_inputs_by_diagnosis(self, mock_input_array, mock_true_array):

        result = get_inputs_by_diagnosis(mock_input_array, mock_true_array)

        np.testing.assert_array_equal(result[0], mock_input_array[[2]])
        np.testing.assert_array_equal(result[1], mock_input_array[[1]])
        np.testing.assert_array_equal(result[2], mock_input_array[[0]])

    @pytest.mark.parametrize(
        "input_idx, expected_result",
        [
            (
                0,
                {
                    "indeterminate": {0: 1, 1: 0, np.nan: 0},
                    "non_epilepsy": {0: 1, 1: 0, np.nan: 0},
                    "epilepsy": {0: 0, 1: 1, np.nan: 0},
                },
            ),
            (
                1,
                {
                    "indeterminate": {0: 0, 1: 0, np.nan: 1},
                    "non_epilepsy": {0: 1, 1: 0, np.nan: 0},
                    "epilepsy": {0: 1, 1: 0, np.nan: 0},
                },
            ),
        ],
    )
    def test_get_input_counts(self, input_idx, expected_result, mock_input_array):

        input_arrays = (
            mock_input_array[[2]],
            mock_input_array[[1]],
            mock_input_array[[0]],
        )
        result = get_input_counts(input_arrays, input_idx)
        assert result == expected_result

    def test_get_metrics(self, mock_input_array, mock_pred_array, mock_true_array):
        result = get_metrics(mock_input_array, mock_pred_array, mock_true_array)
        expected = {
            "Name": "Evaluation 1",
            "Description": "Metrics for Epilepsy vs Non-Epilepsy classes.",
            "Summary": {
                "total": {"predicted": 3, "true": 3},
                "total_classes": {
                    "predicted": 6,
                    "true": 6,
                },
            },
            "Counts": {
                "responses": {},
                "inputs": {
                    0: {
                        "indeterminate": {0: 1, 1: 0, np.nan: 0},
                        "non_epilepsy": {0: 1, 1: 0, np.nan: 0},
                        "epilepsy": {0: 0, 1: 1, np.nan: 0},
                    },
                    1: {
                        "indeterminate": {0: 0, 1: 0, np.nan: 1},
                        "non_epilepsy": {0: 1, 1: 0, np.nan: 0},
                        "epilepsy": {0: 1, 1: 0, np.nan: 0},
                    },
                    2: {
                        "indeterminate": {0: 0, 1: 0, np.nan: 1},
                        "non_epilepsy": {0: 1, 1: 0, np.nan: 0},
                        "epilepsy": {0: 0, 1: 1, np.nan: 0},
                    },
                    3: {
                        "indeterminate": {0: 0, 1: 0, np.nan: 1},
                        "non_epilepsy": {0: 1, 1: 0, np.nan: 0},
                        "epilepsy": {0: 0, 1: 0, np.nan: 1},
                    },
                    4: {
                        "indeterminate": {0: 1, 1: 0, np.nan: 0},
                        "non_epilepsy": {0: 1, 1: 0, np.nan: 0},
                        "epilepsy": {0: 0, 1: 1, np.nan: 0},
                    },
                    5: {
                        "indeterminate": {0: 0, 1: 0, np.nan: 1},
                        "non_epilepsy": {0: 1, 1: 0, np.nan: 0},
                        "epilepsy": {0: 1, 1: 0, np.nan: 0},
                    },
                },
                "diagnoses": {
                    "predicted": {
                        "indeterminate": 1,
                        "non_epilepsy": 1,
                        "epilepsy": 1,
                    },
                    "true": {
                        "indeterminate": 1,
                        "non_epilepsy": 1,
                        "epilepsy": 1,
                    },
                },
            },
            "Performance": {
                "accuracy": {
                    "total": 3,
                    "percentage": 1.0,
                },
                "accuracy_balanced": {"total": 1.0},
            },
        }

        assert result == expected
