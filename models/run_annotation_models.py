from core.models import NERModel
import datetime

# Define the data as global variables
relevant_data_path = "../data/annotated_data/"
corpus_files_path_prefix = relevant_data_path + "data_splits/stratified_entities/"
# TRAIN DATA
train_data_path = corpus_files_path_prefix + "ct_neuro_train_data_787.json"  # "ct_neuro_train_data_713.json"
# TEST DATA
test_data_path = corpus_files_path_prefix + "ct_neuro_test_data_153.json"  # "ct_neuro_test_data_90.json"
test_data_path_csv = corpus_files_path_prefix + "ct_neuro_test_merged_153.csv"  # "ct_neuro_test_merged_90.csv"
# OUTPUT
path_to_save_output_annotations = "./predictions/rebuttal/"

short_to_long_class_names_map = {
    "DRUG": "DRUG",
    "BEH": "BEHAVIOURAL",
    "SURG": "SURGICAL",
    "PHYS": "PHYSICAL",
    "RADIO": "RADIOTHERAPY",
    "OTHER": "OTHER",
    "COND": "CONDITION",
    "CTRL": "CONTROL"
}


def run_inference_hugging_face_model(hugging_face_model_name, hugging_face_model_path, run_BIO_annotations=True):
    model_name_str = "bert-base-uncased"
    if "/" in hugging_face_model_name:
        model_name_str = hugging_face_model_name.split("/")[1]
    model = NERModel("huggingface", hugging_face_model_name, hugging_face_model_path, short_to_long_class_names_map)

    ### ANNOTATE WITH BIO OUTPUT
    if run_BIO_annotations:
        predict_dataset_with_pred = model.bert_predict_bio_format(train_data_path, test_data_path, "tokens",
                                                                  "ner_tags")
        output_path = path_to_save_output_annotations + "ct_neuro_test_annotated_{}_BIO_{}.csv".format(model_name_str,
                                                                                                       current_date)
        predict_dataset_with_pred.to_csv(output_path, sep=",")
        print(f"BIO annotations for {model_name_str} saved in {output_path}.")
    else:
        ### ANNOTATE WITH TUPLE OUTPUT
        annotated_ds = model.annotate(test_data_path_csv, "text")
        output_path = path_to_save_output_annotations + "ct_neuro_test_annotated_{}_{}.csv".format(model_name_str,
                                                                                                   current_date)
        annotated_ds.to_csv(output_path, sep=",")
        print(f"Tuple annotations for {model_name_str} saved in {output_path}.")


if __name__ == '__main__':

    current_date = datetime.datetime.now().strftime("%Y%m%d")

    run_linkbert = False
    run_biobert = True
    run_bert_base_uncased = False

    # TODO: Error "Placeholder storage has not been allocated on MPS device!" when trying to run tuple and BIO annotations sequentially?
    run_BIO_annotations = True  # If set to false the annotations will be saved in tuple format.
    run_regex_dictionary = False

    #### BERT BASE ####
    if run_bert_base_uncased:
        print("Running BERT-BASE model_annotations.")
        hugging_face_model_name = "bert-base-uncased"
        hugging_face_model_path = "./bert/bert-base-uncased/epochs_15_data_size_100_iter_5/"
        run_inference_hugging_face_model(hugging_face_model_name, hugging_face_model_path, run_BIO_annotations)

    #### LinkBERT ####
    if run_linkbert:
        print("Running LinkBERT model_annotations.")
        hugging_face_model_name = "michiyasunaga/BioLinkBERT-base"
        hugging_face_model_path = "./bert/michiyasunaga_biolinkbert/epochs_15_data_size_100_iter_4/"
        run_inference_hugging_face_model(hugging_face_model_name, hugging_face_model_path, run_BIO_annotations)

    #### BioBERT ####
    if run_biobert:
        print("Running BioBERT model_annotations.")
        hugging_face_model_name = "dmis-lab/biobert-v1.1"
        hugging_face_model_path = "./bert/dmis-lab_biobert-v1.1/epochs_15_data_size_100_iter_4/"
        run_inference_hugging_face_model(hugging_face_model_name, hugging_face_model_path, run_BIO_annotations)

    #### RegEx ####
    if run_regex_dictionary:
        print("Running RegEx Dictionary Lookup model_annotations.")
        regex_model = NERModel("regex", "regex")
        annotated_ds = regex_model.annotate(test_data_path_csv, "text")
        annotated_ds.to_csv(
            path_to_save_output_annotations + "ct_neuro_test_annotated_{}_{}.csv".format("regex", current_date),
            sep=",")
