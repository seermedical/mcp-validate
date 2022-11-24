"""
Model adapted from Beniczky S, et. al. A web-based algorithm to rapidly classify seizures for
the purpose of drug selection. Epilepsia. 2021 Oct;62(10):2474-2484. doi: 10.1111/epi.17039.
Epub 2021 Aug 22. PMID: 34420206.

Online tool: epipick.org

Authors: Dominique Eden & Pip Karoly
"""

import argparse
import json
from typing import Mapping, Optional

from src.generate_inputs import transform_input
from src.generate_outputs import get_predicted_output, get_true_output
from src.metrics import get_accuracy


def read_json(path: str) -> Mapping:
    """Reads JSON file and returns a dict.

    Args:
        path: Absolute path to JSON file.

    Returns:
        data: Dictionary of data read from JSON file.
    """

    with open(path, "r") as f:
        data = json.load(f)
    return data


def run(
    input_data_file: str, input_billing_codes_file: str, output_path: Optional[str]
) -> None:

    input_data, input_billing_codes = read_json(input_data_file), read_json(
        input_billing_codes_file
    )

    # Get input array
    input_array = transform_input(input_data)

    # Run model
    predicted_output = get_predicted_output(input_array)
    true_output = get_true_output(input_billing_codes)

    # Get metrics
    accuracy = get_accuracy(predicted_output, true_output, normalize=False)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="EpiPickModelValidation",
        description="Validation of EpiPick Model against Mayo Clinic Platform data.",
    )
    parser.add_argument(
        "--input_data_file",
        required=True,
        help="Path to JSON file storing patient responses.",
    )
    parser.add_argument(
        "--input_billing_codes_file",
        help="Path to JSON file storing patient ICD-10 billing codes.",
    )
    parser.add_argument("-o", "--output_path", help="Path to save outputs.")

    args = parser.parse_args()

    run(args.input_data, args.input_billing_codes, args.output_path)
