from core.models import NERModel
import datetime

if __name__ == '__main__':

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

    current_date = datetime.datetime.now().strftime("%Y%m%d")

    run_linkbert = False
    run_biobert = False
    run_bert_base_uncased = True

    # TODO: Error "Placeholder storage has not been allocated on MPS device!" when trying to run tuple and BIO annotations sequentially?
    run_tuples_annotations = True
    run_BIO_annotations = False
    run_regex_dictionary = False

    relevant_data_path = "../data/annotated_data/"
    corpus_files_path_prefix = relevant_data_path + "data_splits/"
    train_data_path = corpus_files_path_prefix + "ct_neuro_train_data_713.json"
    test_data_path = corpus_files_path_prefix + "ct_neuro_test_data_90.json"
    test_data_path_csv = corpus_files_path_prefix + "ct_neuro_test_merged_90.csv"
    output_annotations_path_prefix = "./predictions/"

    #train_data_path = "/Users/donevas/Desktop/Projects/Univeristy/PhD/Code/NeuroTrialNER/temp/corpus_files/ct_neuro_train_data_713.json"
    #test_data_path = "/Users/donevas/Desktop/Projects/Univeristy/PhD/Code/NeuroTrialNER/temp/corpus_files/ct_neuro_test_data_90.json"

    #### BERT BASE ####
    if run_bert_base_uncased:
        print("Running BERT-BASE model_annotations.")
        hugging_face_model_name = "bert-base-uncased"
        hugging_face_model_path = "./bert/trained/bert-base-uncased/epochs_15_data_size_100_iter_4/"
        model_name_str = "bert-base-uncased"
        model = NERModel("huggingface", hugging_face_model_name, hugging_face_model_path, short_to_long_class_names_map)

        ### ANNOTATE WITH BIO OUTPUT
        if run_BIO_annotations:
            predict_dataset_with_pred = model.bert_predict_bio_format(train_data_path, test_data_path, "tokens",
                                                                      "ner_tags")
            predict_dataset_with_pred.to_csv(
                output_annotations_path_prefix + "ct_neuro_test_annotated_{}_BIO_{}.csv".format(model_name_str,
                                                                                                current_date), sep=",")
        if run_tuples_annotations:
            ### ANNOTATE WITH TUPLE OUTPUT
            annotated_ds = model.annotate(test_data_path_csv, "text")
            annotated_ds.to_csv(
                output_annotations_path_prefix + "ct_neuro_test_annotated_{}_{}.csv".format(model_name_str,
                                                                                            current_date), sep=",")


    #### LinkBERT ####
    if run_linkbert:
        print("Running LinkBERT model_annotations.")
        hugging_face_model_name = "michiyasunaga/BioLinkBERT-base"
        hugging_face_model_path = "./bert/trained/michiyasunaga_biolinkbert/epochs_15_data_size_100_iter_2/"
        model_name_str = hugging_face_model_name.split("/")[1]
        model = NERModel("huggingface", hugging_face_model_name, hugging_face_model_path, short_to_long_class_names_map)

        ### ANNOTATE WITH BIO OUTPUT
        if run_BIO_annotations:
            predict_dataset_with_pred = model.bert_predict_bio_format(train_data_path, test_data_path, "tokens",
                                                                      "ner_tags")
            predict_dataset_with_pred.to_csv(
                output_annotations_path_prefix + "ct_neuro_test_annotated_{}_BIO_{}.csv".format(model_name_str,
                                                                                                current_date), sep=",")
        if run_tuples_annotations:
            ### ANNOTATE WITH TUPLE OUTPUT
            annotated_ds = model.annotate(test_data_path_csv, "text")
            annotated_ds.to_csv(
                output_annotations_path_prefix + "ct_neuro_test_annotated_{}_{}.csv".format(model_name_str,
                                                                                            current_date), sep=",")

    #### BioBERT ####
    if run_biobert:
        print("Running BioBERT model_annotations.")
        hugging_face_model_name = "dmis-lab/biobert-v1.1"
        hugging_face_model_path = "./bert/trained/dmis-lab_biobert_v1.1/epochs_15_data_size_100_iter_2/"
        model_name_str = hugging_face_model_name.split("/")[1]
        model = NERModel("huggingface", hugging_face_model_name, hugging_face_model_path, short_to_long_class_names_map)

        ### ANNOTATE WITH BIO OUTPUT
        if run_BIO_annotations:
            predict_dataset_with_pred = model.bert_predict_bio_format(train_data_path, test_data_path, "tokens",
                                                                      "ner_tags")
            predict_dataset_with_pred.to_csv(
                output_annotations_path_prefix + "ct_neuro_test_annotated_{}_BIO_{}.csv".format(model_name_str,
                                                                                                current_date), sep=",")
        if run_tuples_annotations:
            ### ANNOTATE WITH TUPLE OUTPUT
            annotated_ds = model.annotate(test_data_path_csv, "text")
            annotated_ds.to_csv(
                output_annotations_path_prefix + "ct_neuro_test_annotated_{}_{}.csv".format(model_name_str,
                                                                                            current_date), sep=",")

    #### RegEx ####
    if run_regex_dictionary:
        print("Running RegEx Dictionary Lookup model_annotations.")
        regex_model = NERModel("regex", "regex")
        annotated_ds = regex_model.annotate(test_data_path_csv, "text")
        annotated_ds.to_csv(
            output_annotations_path_prefix + "ct_neuro_test_annotated_{}_{}.csv".format("regex", current_date), sep=",")
