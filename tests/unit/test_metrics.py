import numpy as np
import pytest


from src.metrics import get_inputs_by_diagnosis, get_input_counts, get_metrics


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
