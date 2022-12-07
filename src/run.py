"""
Model adapted from Beniczky S, et. al. A web-based algorithm to rapidly classify seizures for
the purpose of drug selection. Epilepsia. 2021 Oct;62(10):2474-2484. doi: 10.1111/epi.17039.
Epub 2021 Aug 22. PMID: 34420206.

Online tool: epipick.org

Authors: Dominique Eden & Pip Karoly
"""

import argparse
import json
from typing import Dict, Optional

from metrics import get_metrics
from generate_inputs import transform_input
from generate_outputs import get_predicted_output, get_true_output


def read_json(path: str) -> Dict:
    """Reads JSON file and returns a dict.

    Args:
        path: Absolute path to JSON file.

    Returns:
        dict: Dictionary of data read from JSON file.
    """

    with open(path, "r") as f:
        return json.load(f)


def run(
    input_data_file: str, input_billing_codes_file: str, output_path: Optional[str]
) -> None:

    input_data, input_billing_codes = read_json(input_data_file), read_json(
        input_billing_codes_file
    )

    # TODO: add test where questions are equivalent

    # Get input array
    input_array = transform_input(input_data)

    # Run model
    predicted_output = get_predicted_output(input_array)
    true_output = get_true_output(input_billing_codes)

    # Get metrics
    metrics = get_metrics(input_data, predicted_output, true_output, normalize=False)

    return


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
        "-o", "--output_path", default=".", help="Path to save outputs."
    )

    args = parser.parse_args()

    run(args.input_responses_file, args.input_billing_codes_file, args.output_path)
