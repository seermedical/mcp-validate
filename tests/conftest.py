import numpy as np
import pytest


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
        "What other things do you experience right before or at the beginning of a seizure?": response_0,
        "Please describe what you feel right before or at the beginning of a seizure.": response_1,
        "Please specify other warning.": response_2,
        "Please specify other injuries.": response_3,
        "Which warnings do you get before you have a seizure?": response_4,
        "Please specify other symptoms.": response_5,
        "Describe what happens during your seizures.": response_6,
        "How long do your seizures last?": response_7,
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
    return {
        "patient_7": mock_input_dict_template(
            response_0="I get a headache and somtimes a bit dizzy!",
            response_1="Usually when I go to the toilet.",
            response_6="Blacking out.",
        ),
    }


@pytest.fixture
def mock_input_dict():
    return {
        "patient_8": mock_input_dict_template(
            response_0="I go pale, I get a headache.",
            response_6="I droop.",
        ),  # non-epilepsy
        "patient_9": mock_input_dict_template(
            response_0="I'm not sure",
            response_1="I'm not sure",
            response_6="I'm not sure",
            response_7="a few seconds",
        ),  # epilepsy
        "patient_10": mock_input_dict_template(
            response_0="Some text", response_1="Some text"
        ),  # indeterminate
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


@pytest.fixture
def mock_input_billing_codes():
    return {"patient_8": [], "patient_9": [], "patient_10": []}
