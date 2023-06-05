import numpy as np
import pytest
from src.generate_outputs import set_diagnosis


class TestSetDiagnosis:
    """Tests function set_diagnosis() to check input diagnoses are
    correctly categorised in the true output array.
    Elements are represented as 0 = negative diagnosis, or 1 = positive diagnosis. N.b. A
    patient may have multiple diagnoses.
    # Example:
        # +--------------+--------------+----------+-------+-------------+---------+
        # indeterminate  | non_epilepsy | epilepsy | focal | generalized | unknown |
        # +--------------+--------------+----------+-------+-------------+---------+
        # | 1            | 0            | 0        | 0     | 0           | 0       |
        # +--------------+--------------+----------+-------+-------------+---------+
    """

    @pytest.mark.parametrize(
        "mock_patient_codes, expected_result",
        [
            (["R55"], np.array([[0, 1, 0, 0, 0, 0]])),  # Tests syncope for non-epilepsy
            (
                ["G43.119"],
                np.array([[0, 1, 0, 0, 0, 0]]),
            ),  # Tests paroxysmal for non-epilepsy
            (
                ["G40.0"],
                np.array([[0, 0, 1, 1, 0, 0]]),  # Tests focal for epilepsy
            ),  # Tests
            (
                ["G40.813"],
                np.array([[0, 0, 1, 0, 0, 1]]),  # Tests unknown for epilepsy
            ),  # Tests
            (
                ["A40.1"],
                np.array([[1, 0, 0, 0, 0, 0]]),
            ),  # Tests indeterminate
            (
                [],
                np.array([[1, 0, 0, 0, 0, 0]]),
            ),  # Tests indeterminate
        ],
    )
    def test_set_diagnosis(self, mock_patient_codes, expected_result):
        """Tests the function that sets a patient's diagnosis."""

        result = set_diagnosis(patient_codes=mock_patient_codes)

        np.testing.assert_array_equal(result, expected_result)
