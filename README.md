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
```bib
to_prodigy_data_converter.py
```
The code in to_prodigy_data_converter.py converts the extracted sample from the AACT database (folder /raw_data_aact_sample) from .csv to a .json file (folder ./data_for_prodigy). This data will be used for annotation in prodigy.
During our first annotation period, we first annotated 100 pilot examples and refined the annotation guidelines. Then we annotated another 400 examples. In a second annotation period, two annotators had another 500 samples to annotate.
The instructions for using prodigy are in the file "Using Prodigy for Named Entity Annotation.docx" in the ./data_for_prodigy folder. The resulting annotations are stored in ./annotation_round_1 and ./annotation_round_2. There you can find the individual annotations of each annotator. 
Those outputs were then further reviewed in prodigy in order to resolve conflicts and create the final datasets. The resulting datasets
are the two .jsonl files "neuro_merged_all_433" and "neuro_merged_annotations_405_2batch".
```bib
from_prodigy_data_converter.py
```
In the from_prodigy_data_converter.py file only the merged annotations are kept, i.e. after review. The resulting files are 
"ct_neuro_final_target_annotated_ds_round_1.jsonl" and "ct_neuro_405_target_annotated_ds_round_2.jsonl".
