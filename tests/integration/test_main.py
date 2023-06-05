import os
import numpy as np
from src.run import run, read_json

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


class TestEndToEnd:
    """Runs end-to-end tests."""

    def test_end_to_end(self):
        run(
            input_data_file=os.path.join(TEST_DATA_DIR, "test_responses.json"),
            input_billing_codes_file=os.path.join(
                TEST_DATA_DIR, "test_billing_codes.json"
            ),
            output_path=TEST_OUTPUT_DIR,
        )

        input_array = np.load(os.path.join(TEST_OUTPUT_DIR, "input_array.npy"))
        pred_output = np.load(os.path.join(TEST_OUTPUT_DIR, "pred_output.npy"))
        true_output = np.load(os.path.join(TEST_OUTPUT_DIR, "true_output.npy"))
        metrics = read_json(os.path.join(TEST_OUTPUT_DIR, "metrics.json"))

        expected_input_array = np.array(
            (
                [np.nan] * 6,
                [0, 0, 0, np.nan, 0, 0],
                [1, 1, 0, np.nan, 1, 0],
                [1, 1, 0, np.nan, 1, 0],
                [0, 0, 0, np.nan, 0, 0],
            )
        )
        expected_pred_output = np.array(
            (
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                ]
            )
        )
        expected_true_output = np.array(
            (
                [
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                ]
            )
        )

        expected_metrics = {
            "Name": "Evaluation 1",
            "Description": "Metrics for Epilepsy vs Non-Epilepsy classes.",
            "Summary": {
                "total": {"predicted": 5.0, "true": 5.0},
                "total_classes": {
                    "predicted": 6.0,
                    "true": 6.0,
                },
            },
            "Counts": {
                "responses": {},
                "inputs": {
                    "0": {
                        "indeterminate": {"0": 0, "1": 0, "NaN": 0},
                        "non_epilepsy": {"0": 1, "1": 1, "NaN": 1},
                        "epilepsy": {"0": 1, "1": 1, "NaN": 0},
                    },
                    "1": {
                        "indeterminate": {"0": 0, "1": 0, "NaN": 0},
                        "non_epilepsy": {"0": 1, "1": 1, "NaN": 1},
                        "epilepsy": {"0": 1, "1": 1, "NaN": 0},
                    },
                    "2": {
                        "indeterminate": {"0": 0, "1": 0, "NaN": 0},
                        "non_epilepsy": {"0": 2, "1": 0, "NaN": 1},
                        "epilepsy": {"0": 2, "1": 0, "NaN": 0},
                    },
                    "3": {
                        "indeterminate": {"0": 0, "1": 0, "NaN": 0},
                        "non_epilepsy": {"0": 0, "1": 0, "NaN": 3},
                        "epilepsy": {"0": 0, "1": 0, "NaN": 2},
                    },
                    "4": {
                        "indeterminate": {"0": 0, "1": 0, "NaN": 0},
                        "non_epilepsy": {"0": 1, "1": 1, "NaN": 1},
                        "epilepsy": {"0": 1, "1": 1, "NaN": 0},
                    },
                    "5": {
                        "indeterminate": {"0": 0, "1": 0, "NaN": 0},
                        "non_epilepsy": {"0": 2, "1": 0, "NaN": 1},
                        "epilepsy": {"0": 2, "1": 0, "NaN": 0},
                    },
                },
                "diagnoses": {
                    "predicted": {
                        "indeterminate": 1.0,
                        "non_epilepsy": 2.0,
                        "epilepsy": 2.0,
                    },
                    "true": {
                        "indeterminate": 0.0,
                        "non_epilepsy": 3.0,
                        "epilepsy": 2.0,
                    },
                },
            },
            "Performance": {
                "accuracy": {
                    "total": 2.0,
                    "percentage": 0.5,
                },
                "accuracy_balanced": {"total": 0.5},
            },
        }
        np.testing.assert_array_equal(input_array, expected_input_array)
        np.testing.assert_array_equal(true_output, expected_true_output)
        np.testing.assert_array_equal(pred_output, expected_pred_output)

        assert expected_metrics == metrics
