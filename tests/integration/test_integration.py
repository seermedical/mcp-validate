import pytest


@pytest.fixture
def mock_dict():
    return {
        'patient_id_1': {
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
        },
        'patient_id_2': {
            'What other things do you experience right before or at the beginning of a seizure?':
            'Response 8',
            'Please describe what you feel right before or at the beginning of a seizure.':
            'Response 9',
            'Please specify other warning.':
            '',
            'Please specify other injuries.':
            '',
            'What injuries have you experienced during a seizure?s':
            'Response 10',
            'Please specify other symptoms.':
            'Response 11',
            'Describe what happens during your seizures.':
            'Response 12',
            'Please describe what you feel right before or at the beginning of a seizure.':
            'Response 13',
            'How long do your seizures last?':
            '7 - 5 minutes'
        }    }
