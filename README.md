# NeuroTrialNER
Complementary code for "NeuroTrialNER: An Annotated Corpus for Neurological Diseases and Therapies in Clinical Trial Registries".
This repo provides the code & data of our paper: <TODO: insert link>. Please note that ChatGPT has assisted with parts of the code, usually for the generation of the visualizations. The implementation of the BERT-based training was based on https://github.com/michiyasunaga/LinkBERT. 

# 1. Set up the environment
The project is build using poetry for dependency management. Instructions on how to install poetry can be found in the [documentation](https://python-poetry.org/docs/).  
To install the defined dependencies for the project, make sure you have the .toml and .lock files and run the _install_ command.
```bib
poetry install
```
The pyproject.toml file contains all relevant packages that need to be installed for running the project. The poetry.lock file is needed to ensure the same version of the installed libraries.

# 2. Data
The code related to the data processing and annotation steps is in the folder ./data.

## Terminology Dictionary Generation
### Neuropsychiatric Disease Names
Prerequisites to reproduce the results:
- Download the latest MeSH terminology dump from [NIH mesh download](https://www.nlm.nih.gov/databases/download/mesh.html). For the current project the 2023 year was used (mesh_desc2023.xml). Place this file in the ./data/neuro_diseases_terminology/input/ folder.
- In order to be able to use the ICD APIs, first you need to create an account on the [ICD API Home page](https://icd.who.int/icdapi). You need to put the client_id_ICD and client_secret_ICD in your local credentials.txt file.

```bib
extract_disease_names_from_icd_api.py
```
The code from this file traverses the relevant starting nodes from the ICD-11 API and extracts all disease names below them. The starting points used
are ["Diseases of the nervous system"](https://icd.who.int/browse11/l-m/en#/http%3a%2f%2fid.who.int%2ficd%2fentity%2f1296093776) and ["Mental, behavioural or neurodevelopmental disorders"](https://icd.who.int/browse11/l-m/en#/http%3a%2f%2fid.who.int%2ficd%2fentity%2f334423054).
Note that it can take up to 10 minutes for the data to be collected.

```bib
extract_disease_names_and_synonyms_from_mesh_dump.py
```
With this code we load the MeSH dump and filter it for the relevant diseases based on the [Neurology_disease-list_MeSH.xlsx](data%2Fneuro_diseases_terminology%2Finput%2FNeurology_disease-list_MeSH.xlsx).

```bib
merge_mesh_and_icd_terminology.py
```
With the functions here we merge the two disease lists from MeSH and ICD ([diseases_dictionary_mesh_icd.csv](data%2Fneuro_diseases_terminology%2Foutput%2Fdiseases_dictionary_mesh_icd.csv)). Furthermore, we generate a flat list ([diseases_dictionary_mesh_icd_flat.csv](data%2Fneuro_diseases_terminology%2Foutput%2Fdiseases_dictionary_mesh_icd_flat.csv))
that contains all synonyms and spelling variations on a new line. This flat list was used to filter the AACT database as described in the next section.

### Drug Names
The files to compile the drug dictionary are in [drug_names_terminology](data%2Fdrug_names_terminology). They were directly used by
the code described in [NER Methods/Dictionary Lookup](#dictionary-lookup) to annotate for drugs.

## Sample Extraction from AACT
For our project, a static copy of the AACT database was downloaded on May 12 2023. Following the [installation instructions](https://aact.ctti-clinicaltrials.org/snapshots), 
a local PostgreSQL database was populated from the database file, comprising 451’860 unique trials. 

The _conditions_ table was joined with our disease list and we kept only those trials. 
This resulted in 40’842 unique trials related to neurological conditions (of which 35’969 were registered as interventional trials).
The official title (from table _ctgov.studies_) of each trial together with its short description (from table _ctgov.brief_summaries_) was extracted to a csv file and prepared for annotation.

For our third annotation round, we wanted to sample only non-drug interventions. The filtering for this was made in the AACT database
by excluding intervention_type values equal to DRUG, BIOLOGICAL or DIETARY. To make sure that no overlaps with our existing samples exist,
we extracted the AACT data and filtered it as shown in the notebook [Rebuttal_ACL_Additional_Data_Sampling.ipynb](data%2FRebuttal_ACL_Additional_Data_Sampling.ipynb).

## Annotation with Prodigy
IMPORTANT: The annotations guidelines can be found here [annotation_guidelines](annotation_guidelines).

```bib
to_prodigy_data_converter.py
```
The code in to_prodigy_data_converter.py converts the extracted sample from the AACT database [data_aact_sample](data%2Fdata_aact_sample)from .csv to a .json file ([data_for_prodigy](data%2Fdata_for_prodigy)). 
This data will be used as input for annotation in prodigy.
During our first annotation period, we first annotated 100 pilot examples and refined the annotation guidelines. 
Then we annotated another 400 examples. In a second annotation period, two annotators had another 500 samples to annotate. Finally, in a third annotation round
we annotated 200 non-drug intervention trials.

Below you see the Prodigy command we used to start an annotation session in Prodigy. For further instructions for using prodigy see the file [Using Prodigy for Named Entity Annotation.docx](data%2Fdata_for_prodigy%2FUsing%20Prodigy%20for%20Named%20Entity%20Annotation.docx). 

```
prodigy ner.manual <PRODIGY_DATASET_NAME> blank:en ./<ANNOTATION_DATA_FILE_NAME>.jsonl --label DRUG,BEHAVIOURAL,SURGERY,DIET,RADIOTHERAPY,REHABILITATION,I-OTHER,DISEASE
```
The resulting annotations are stored in [annotation_round_1](data%2Fannotated_data%2Fannotation_round_1), [annotation_round_2](data%2Fannotated_data%2Fannotation_round_2), and [annotation_round_3](data%2Fannotated_data%2Fannotation_round_3).
There you can find the individual annotations with all prodigy extracted details of each annotator (file names start with the annotator code names, i.e, bvi, sed or aec). 
Those outputs were then further reviewed in prodigy in order to resolve conflicts and create the final datasets. 
The resulting merged datasets from each annotation round are the .jsonl files: 
- round 1: [neuro_merged_all_433.jsonl](data%2Fannotated_data%2Fannotation_round_1%2Fneuro_merged_all_433.jsonl)
- round 2: [neuro_merged_annotations_405_2batch.jsonl](data%2Fannotated_data%2Fannotation_round_2%2Fneuro_merged_annotations_405_2batch.jsonl)
- round 3: [neuro_merged_annotations_nondrug_60_batch_3.jsonl](data%2Fannotated_data%2Fannotation_round_3%2Fneuro_merged_annotations_nondrug_60_batch_3.jsonl) and [neuro_merged_annotations_nondrug_143_batch_3.jsonl](data%2Fannotated_data%2Fannotation_round_3%2Fneuro_merged_annotations_nondrug_143_batch_3.jsonl)

```bib
from_prodigy_data_converter.py
```
In from_prodigy_data_converter.py we filter the merged annotations from the Prodigy review session to keep only the agreed on annotations. Note that per default,
the Prodigy "review" recipe stores the annotations from all reviewers and in addition the outputs from the resolution session.

The resulting files are:
- round 1:[ct_neuro_final_target_annotated_ds_round_1.jsonl](data%2Fannotated_data%2Fannotation_round_1%2Fct_neuro_final_target_annotated_ds_round_1.jsonl) 
- round 2:  [ct_neuro_405_target_annotated_ds_round_2.jsonl](data%2Fannotated_data%2Fannotation_round_2%2Fct_neuro_405_target_annotated_ds_round_2.jsonl).
- round 3: [ct_neuro_60_target_annotated_ds_round_3.jsonl](data%2Fannotated_data%2Fannotation_round_3%2Fct_neuro_60_target_annotated_ds_round_3.jsonl) and [ct_neuro_143_target_annotated_ds_round_3.jsonl](data%2Fannotated_data%2Fannotation_round_3%2Fct_neuro_143_target_annotated_ds_round_3.jsonl)

### Inter-Annotator Agreement
```bib
annotation_agreement_evaluation.py
```
This file contains the code to evaluate the annotation agreement using the Cohen Kappa statistics. The score calculation code is adapted
from [rowannicholls](https://rowannicholls.github.io/python/statistics/agreement/cohens_kappa.html). The results are printed out and a confusion matrix is produced
for each annotator pair in [annotations_confusion_matrix](data%2Fannotated_data%2Fcorpus_stats%2Fannotations_confusion_matrix).
### Converting Prodigy annotations to BIO format
```bib
convert_to_bio_and_generate_dataset_split.py
```
In code file we convert the prodigy output annotations to BIO format. For each array of tokens (as split by Prodigy), a corresponding array
of annotations will be generated: "O" for no label, "B-XX" for begin label XX, or "I-XX" inside label XX. 
The resulting full datasets are available in several formats:
- the fields relevant for training: [ct_neuro_final_target_annotated_ds_bio_format_1093.jsonl](data%2Fannotated_data%2Ffinal_combined%2Fct_neuro_final_target_annotated_ds_bio_format_1093.jsonl)

| JSON Key Name | Value Data Type | Description                                                                                                                               |
|---------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| tokens        | List of Strings | A list of tokens (words or punctuation) extracted from the text.                                                                          |
| ner_tags      | List of Strings | Corresponding Named Entity Recognition (NER) tags for each token. Each tag indicates the entity type (e.g., B-OTHER, I-OTHER, O, B-COND). |
| id            | String          | A unique identifier for the data entry, here the clinical trial ID (e.g., NCT02632279).                                                   |- additional prodigy details

- all the details from prodigy: [ct_neuro_final_target_annotated_ds_combined_rounds.jsonl](data%2Fannotated_data%2Ffinal_combined%2Fct_neuro_final_target_annotated_ds_combined_rounds.jsonl) and 
[ct_neuro_final_target_annotated_ds_combined_rounds.csv](data%2Fannotated_data%2Ffinal_combined%2Fct_neuro_final_target_annotated_ds_combined_rounds.csv). Below is 
the metadata for the json file.

| JSON Key Name | Value Data Type | Description                                                                                                                                                                                                                                                                                |
|---------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| nct_id        | String          | A unique identifier for the clinical trial or study (e.g., NCT02632279).                                                                                                                                                                                                                   |
| source        | String          | The source or field from which the text was extracted (e.g., OfficialTitle+BriefSummary).                                                                                                                                                                                                  |
| text          | String          | The full text content extracted from the specified source.                                                                                                                                                                                                                                 |
| tokens        | List of Objects | A list of token objects, where each object contains details about individual tokens in the text. E.g. {"text": "Tryptophan", "start": 0, "end": 10, "id": 0, "ws": true}, {"text": "Depletion", "start": 11, "end": 20, "id": 1, "ws": true}...}                                           |
| spans         | List of Objects | A list of span objects, where each object represents a labeled span of text with start and end positions. E.g. "spans": [{"start": 0, "end": 20, "token_start": 0, "token_end": 1, "label": "OTHER"}, {"start": 24, "end": 43, "token_start": 3, "token_end": 5, "label": "CONDITION"}...] |
| _timestamp    | Integer         | A timestamp indicating when the data was processed or recorded (in Unix epoch format).                                                                                                                                                                                                     |

The final dataset file will be split into train, dev, and test parts. Two types of splits are available: 
- Proportion 80-10-10: The final data used for training the models can be found in [data/data_splits](data%2Fannotated_data%2Fdata_splits). 
- Stratified to balance minority entity types: The final data used for training the models can be found in [data/data_splits/stratified_entities](data%2Fannotated_data%2Fdata_splits%2Fstratified_entities).

### Generating corpus statistics
```bib
generate_corpus_statistics.py
```
Generates information about the total number of entities for each entity type in the datasets. Also outputs the frequency of individual entities. 
Outputs are saved in [corpus_stats](data%2Fannotated_data%2Fcorpus_stats). This data is used in [Generate_Corpus_Stats.ipynb](data%2FGenerate_Corpus_Stats.ipynb) to create visuals of top entities based on their frequency.


# 3. NER Methods
## BERT Models
### Training
The bash script used for fine-tuning a BERT model from HuggingFace on our university computing server is in
[models/bert/run_single_experiment.sh](models%2Fbert%2Frun_single_experiment.sh). 
It executes [models/bert/train_script.py](models%2Fbert%2Ftrain_script.py) via the following command:
```bib
python train_script.py \
    --output_path "../clinical_trials_out/arr_rebuttal/train_size_impact" \
    --model_name_or_path "dmis-lab/biobert-v1.1" \
    --train_data_path "./data/ct_neuro_train_data_787.json" \
    --val_data_path "./data/ct_neuro_dev_data_153.json" \
    --test_data_path "./data/ct_neuro_test_data_153.json" \
    --n_epochs 15 --percentage "$percentage" --i "$i"
```
Please make sure you have the reference to the folder with the data correctly. The three parameters that can be changed are:
- model_name_or_path (default 'michiyasunaga/BioLinkBERT-base'): reference to a local model or a model hosted on huggingface; the other models we used are "dmis-lab/biobert-v1.1" and "bert-base-uncased".
- n_epochs (default 20): the number of epochs for training; our experiments showed that there was no impact on the dev learning curve after more than 10 epochs
- percentage (default 100): percentage value determining how much of the training dataset should be used.

We also used a script that parallelizes the training of each model across separate GPUs, provided in
[models/bert/run_parallel_experiment.sh](models%2Fbert%2Frun_parallel_experiment.sh).

Please note that [wandb](https://docs.wandb.ai/guides/integrations/huggingface) was used to collect and visualize the training results.

### Prediction
```bib
models/run_annotation_models.py
```
This file will initialize the requested models and do prediction on the test dataset. A prerequisite is to have the trained model files available.
The prediction code is in the core module file [models.py](core%2Fmodels.py).

To initialize a BERT model the following command is used:
```bib
 model = NERModel("huggingface", hugging_face_model_name, hugging_face_model_path, short_to_long_class_names_map)

```
Two outputs can be generated. With model.bert_predict_bio_format() an array of the entity class for each token is saved. With model.annotate() the annotations are saved
in the format (start index, end index, type, entity tokens), i.e., (99, 109, 'DRUG', 'Gabapentin'). The results are saved under [predictions](models%2Fpredictions).


## GPT Model
The extraction of condition and intervention using GPT is in [Annotate with GPT.ipynb](models%2Fgpt%2FAnnotate%20with%20GPT.ipynb). Note that the code expects a valid OpenAPI key that
can be read from the credentials.txt file. Note that the annotation can take up to 20 minutes. The results are saved under [predictions](models%2Fpredictions).

## Dictionary Lookup
```bib
models/run_annotation_models.py
```
The dictionary lookup model can be initialized as:

```bib
regex_model = NERModel("regex", "regex")
annotated_ds = regex_model.annotate(test_data_path_csv, "text")
```
The code for the annotation is in [drugs_condition_dictionary_finder.py](core%2Fdrugs_condition_dictionary_finder.py).
What the code does is to first create two dictionaries for the drug and disease names. The dictionary contains all synonyms and spelling variations
from different terminology libraries. To annotate the tokens, there will be two lookups of this dictionaries - of a single token and of two sequential tokens.
If those are found in the dictionaries that token(s) recieve the CONDITION or DRUG label. The code is an expansion of this package - [Drug named entity recognition Python library by Fast Data Science
](https://pypi.org/project/drug-named-entity-recognition/#:~:text=%F0%9F%92%8A%20Drug%20named%20entity%20recognition&text=This%20is%20a%20lightweight%20Python,t%20support%20misspellings%20at%20present.).

## Evaluation

### Token- and Abstract-level
```bib
models/evaluate.py
```
This file will call the implementation of the sequeval, i.e., token-wise, evaluation of the models's performance. The implementation of the evaluation code is in [performance_evaluation.py](core%2Fperformance_evaluation.py).

```bib
models/CT Models Evaluation.ipynb
```
In this notebook, the aggregation of labels on trial level is performed, as well as the normalization of the entities based on the drug and disease dictionaries.
The performance is subsequently evaluated on trial/abstract level.

### Error Analysis
The tool PowerBI was used for the detailed error analysis. A free desktop version can be downloaded from 
[Microsoft Power BI Desktop](https://powerbi.microsoft.com/en-us/downloads/) to open the file [Error_Analysis_PowerBI.pbix](models%2FError_Analysis_PowerBI.pbix).
