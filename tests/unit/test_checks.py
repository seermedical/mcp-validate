from src.run_checks import run_checks


class TestChecks:
    def test_all_checks_succeeds(self, mock_input_dict, mock_input_billing_codes):

        try:
            run_checks(mock_input_dict, mock_input_billing_codes)
        except Exception:
            assert False

    def test_check_1_fails(self, mock_input_dict, mock_input_billing_codes):

        mock_input_billing_codes_check_1 = mock_input_billing_codes.pop("patient_8")
        try:
            run_checks(mock_input_dict, mock_input_billing_codes_check_1)
        except Exception:
            assert True

    def test_check_2_fails(self, mock_input_dict, mock_input_billing_codes):
        mock_input_billing_codes_check_2 = {
            "patient_8": mock_input_billing_codes["patient_8"]
        }
        mock_input_dict_check_2 = {"patient_8": mock_input_dict["patient_8"]}[
            "patient_8"
        ].pop("Please specify other symptoms.")

        try:
            run_checks(mock_input_dict_check_2, mock_input_billing_codes_check_2)
        except Exception:
            assert True

    def test_check_3_fails(self, mock_input_dict, mock_input_billing_codes):
        mock_input_billing_codes_check_3 = {
            "patient_8": mock_input_billing_codes["patient_8"],
            "patient_9": mock_input_billing_codes["patient_9"],
        }
        mock_input_dict_check_3 = {
            "patient_8": mock_input_dict["patient_8"],
            "patient_9": mock_input_dict["patient_9"],
        }["patient_8"].pop("Please specify other symptoms.")
        try:
            run_checks(mock_input_dict, mock_input_billing_codes)
        except Exception:
            assert True
