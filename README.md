# mcp-validate
Development of EpiPick model for validation via Mayo Clinic Platform_Validate program. This model will be used to validate the model published by Sperling's group at Jefferson Comprehensive Epilepsy Center, using Mayo Clinic's Platform data.

Link to paper: https://pubmed.ncbi.nlm.nih.gov/32697354/

To run the code, please ensure the `src` directory is available in the host environment. If run from command line, the model can be run by:

`python run.py <path to JSON file> <path to JSON file>`

For example:

`python run.py patient_responses.json patient_diagnoses.json`
