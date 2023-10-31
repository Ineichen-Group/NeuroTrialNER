# NeuroTrialNER
Complementary code for "NeuroTrialNER: An Annotated Corpus for Neurological Diseases and Therapies in Clinical Trials".
This repo provides the model, code & data of our paper: <TODO: insert link>.

# 1. Set up the environment
The project is build using poetry for dependency management. Instrucitons on how to install poetry can be found in the [documentation](https://python-poetry.org/docs/).  
To install the defined dependencies for the project, make sure you have the .toml and .lock files and run the install command.
```bib
poetry install
```
The .toml file contains all relevant packages that need to be installed for running the project.

# 2. Annotated Data
The code related to the data processing and annotation steps is in the folder ./data.
## Annotation with Prodigy
The file to_prodigy_data_converter.py converts the extracted rows from the AACT database from .csv to a .json file, that will be used for annotation in prodigy.

