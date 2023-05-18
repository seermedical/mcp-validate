"""
Model adapted from Beniczky S, et. al. A web-based algorithm to rapidly classify seizures for
the purpose of drug selection. Epilepsia. 2021 Oct;62(10):2474-2484. doi: 10.1111/epi.17039.
Epub 2021 Aug 22. PMID: 34420206.

Online tool: epipick.org

Authors: Dominique Eden & Pip Karoly
"""

import argparse
import json
import os
from typing import Dict, Union
import numpy as np

from src.generate_inputs import transform_input
from src.generate_outputs import get_predicted_output, get_true_output
from src.metrics import get_metrics
from src.run_checks import run_checks


def read_json(path: str) -> Dict:
    """Reads JSON file and returns a dict.

    Args:
        path: Path to JSON file.

    Returns:
        dict: Dictionary of data read from JSON file.
    """

    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: Dict):
    """Writes JSON file.

    Args:
        path: Path to JSON file.
        data: Data to write.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, allow_nan=True, indent=4)
    return


def save_outputs(outputs: Dict[str, Union[np.ndarray, Dict]], output_path: str):
    """Saves output data to files.

    Args:
        outputs: Dict of key, value pairs indicating name of file and data
            respectively.
        output_path: Directory to save outputs.

    Raises:
        TypeError: _description_
    """
    for k, v in outputs.items():
        if isinstance(v, np.ndarray):
            np.save(os.path.join(output_path, f"{k}.npy"), v)

        elif isinstance(v, dict):
            write_json(os.path.join(output_path, f"{k}.json"), v)

        else:
            raise TypeError("Output type(s) not as expected.")

        print(f"Saved {k} to {output_path}")


def run(input_data_file: str, input_billing_codes_file: str, output_path: str) -> None:

    # Create output folder if doesn't exist
    os.makedirs(output_path, exist_ok=True)

    input_data, input_billing_codes = read_json(input_data_file), read_json(
        input_billing_codes_file
    )

    # Run checks on input data
    run_checks(input_data, input_billing_codes)

    # Get input array
    input_array = transform_input(input_data)

    # Run model
    pred_output = get_predicted_output(input_array)
    true_output = get_true_output(input_billing_codes)

    # Get metrics
    metrics = get_metrics(input_array, pred_output, true_output)

    # Save
    save_outputs(
        outputs={
            "input_array": input_array,
            "pred_output": pred_output,
            "true_output": true_output,
            "metrics": metrics,
        },
        output_path=output_path,
    )

    return print("Complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="EpiPickModelValidation",
        description="Validation of EpiPick Model against Mayo Clinic Platform data.",
    )
    parser.add_argument(
        "input_responses_file",
        help="Path to JSON file storing patient responses.",
    )
    parser.add_argument(
        "input_billing_codes_file",
        help="Path to JSON file storing patient ICD-10 billing codes.",
    )
    parser.add_argument(
        "-o", "--output_path", default=os.getcwd(), help="Path to save outputs."
    )

    args = parser.parse_args()

    run(args.input_responses_file, args.input_billing_codes_file, args.output_path)
