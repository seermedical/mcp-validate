import pytest

from src.generate_inputs import get_input_value, FLAG_1_KEYWORDS


@pytest.fixture
def patient_1_dict():
    return {
        'What other things do you experience right before or at the beginning of a seizure?':
        'Response 1',
        'Please describe what you feel right before or at the beginning of a seizure.':
        'Response 2',
        'Please specify other warning.':
        '',
        'Please specify other injuries.':
        '',
        'What injuries have you experienced during a seizure?s':
        'Response 3',
        'Please specify other symptoms.':
        'Response 4',
        'Describe what happens during your seizures.':
        'Response 5',
        'Please describe what you feel right before or at the beginning of a seizure.':
        'Response 6',
        'How long do your seizures last?':
        'a few seconds'
    }


@pytest.fixture
def patient_2_dict():
    return {
        'What other things do you experience right before or at the beginning of a seizure?':
        'Response 1',
        'Please describe what you feel right before or at the beginning of a seizure.':
        'Response 2',
        'Please specify other warning.':
        '',
        'Please specify other injuries.':
        '',
        'What injuries have you experienced during a seizure?s':
        'Response 3',
        'Please specify other symptoms.':
        'Response 4',
        'Describe what happens during your seizures.':
        'Response 5',
        'Please describe what you feel right before or at the beginning of a seizure.':
        'Response 6',
        'How long do your seizures last?':
        'a few seconds'
    }


class TestGetInputValues:
    """_summary_
    """
    @pytest.mark.parametrize("patient_dict", "expected_boolean",
                             [(patient_1_dict, True)],
                             [(patient_2_dict, False)])
    def test_get_input_values(self, mock_patient_dict, expected_boolean):
        result = get_input_value(input_dict=patient_dict,
                                 list_of_keywords=FLAG_1_KEYWORDS)

        assert result == expected_boolean
