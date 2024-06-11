import pandas as pd
import numpy as np
from core.performance_evaluation import ModelEvaluator
import csv
import ast
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import datetime


# Function to aggregate the evaluation dictionary
def aggregate_evaluation(group):
    evaluation_agg = {
        'tp': sum(entry['tp'] for entry in group['evaluation']),
        'fp': sum(entry['fp'] for entry in group['evaluation']),
        'fn': sum(entry['fn'] for entry in group['evaluation'])
    }
    return evaluation_agg


# Function to convert string representation of dictionary to actual dictionary
def parse_evaluation(evaluation_str):
    return ast.literal_eval(evaluation_str)


def evaluate_entity_class_level_performance():
    entity_level_eval = pd.read_csv(
        "corpus/first_annotation_round/model_evaluation/clintrials_evaluation_entities_stats_linkbert.csv")
    entity_level_eval['evaluation'] = entity_level_eval['evaluation'].apply(parse_evaluation)
    # Group by 'entity_class' and apply the aggregate_evaluation function
    aggregated_df = entity_level_eval.groupby('entity_class').apply(aggregate_evaluation).reset_index()
    aggregated_df.rename(columns={0: 'confusion_matrix'}, inplace=True)

    for index, row in aggregated_df.iterrows():
        entity_class = row['entity_class']
        confusion_matrix = row['confusion_matrix']
        tp = confusion_matrix['tp']
        fp = confusion_matrix['fp']
        fn = confusion_matrix['fn']
        tn = 0
        evaluator = ModelEvaluator()
        print(entity_class)
        print(evaluator.evaluate_confusion_matrix([tp, fp, fn, tn]))


if __name__ == '__main__':
    # evaluate_entity_class_level_performance()

    # exit()

    target_dataset = "clintrials"
    run_linkbert = False
    run_biobert = True
    run_regex = False
    run_basebert = False

    train_entities_distr_df = pd.read_csv("../data/annotated_data/corpus_stats/clintrials_train_entities_stats.csv")
    train_entities_distr_dict = train_entities_distr_df.set_index('entity_token').to_dict(orient='index')
    train_data_file = "../data/annotated_data/data_splits/stratified_entities/ct_neuro_train_data_787.json"

    bert_annotated_files_path_prefix = "./predictions/rebuttal/bert/"
    output_evaluation_path_prefix = "./evaluations/"

    if run_basebert:
        print("Eval BERT-base model...")
        hugging_face_model_name = "bert-base-uncased"
        model_name_str = "bert-base-uncased"
        annotated_data_path = bert_annotated_files_path_prefix + "ct_neuro_test_annotated_{}_{}.csv".format(model_name_str, "20240131")
        annotated_data_path_bio = bert_annotated_files_path_prefix + "ct_neuro_test_annotated_{}_BIO_{}.csv".format(model_name_str, "20240131")
        df = pd.read_csv(annotated_data_path)
        evaluator = ModelEvaluator(target_dataset, df, source_id_col_name="nct_id", source_sent_col_name='text',
                                   train_entities_distr_dict=train_entities_distr_dict,
                                   confidence=0.95, output_file_path=output_evaluation_path_prefix)

        print("*** Evaluating BIO format - ENTITY level***")
        print(evaluator.evaluate_bert_bio(annotated_data_path_bio, train_data_file,
                                          "ner_tags", return_format="all", target_labels_column="labels",
                                          predicted_labels_column=f"predictions_{model_name_str}"))
        print("*** Evaluating exact match format ***")
        #print(evaluator.evaluate('ner_manual_ct_target', f'ner_prediction_{model_name_str}_normalized', model_name_str, ignore_labels_for_evaluation=False))

    if run_linkbert:
        print("Eval LinkBERT model...")
        hugging_face_model_name = "michiyasunaga/BioLinkBERT-base"
        model_name_str = hugging_face_model_name.split("/")[1]
        annotated_data_path = bert_annotated_files_path_prefix + "ct_neuro_test_annotated_{}_{}.csv".format(model_name_str, "20230916")
        annotated_data_path_bio = bert_annotated_files_path_prefix + "ct_neuro_test_annotated_{}_BIO_{}.csv".format(model_name_str, "20230916")
        df = pd.read_csv(annotated_data_path)
        evaluator = ModelEvaluator(target_dataset, df, source_id_col_name="nct_id", source_sent_col_name='text',
                                   train_entities_distr_dict=train_entities_distr_dict,
                                   confidence=0.95, output_file_path=output_evaluation_path_prefix)

        print("*** Evaluating BIO format - ENTITY level***")
        print(evaluator.evaluate_bert_bio(annotated_data_path_bio, train_data_file,
                                          "ner_tags", return_format="all", target_labels_column="labels",
                                          predicted_labels_column=f"predictions_{model_name_str}"))
        print("*** Evaluating exact match format ***")
        print(evaluator.evaluate('ner_manual_ct_target', f'ner_prediction_{model_name_str}_normalized', model_name_str, ignore_labels_for_evaluation=False))

    if run_biobert:
        print("Eval BioBERT model...")
        hugging_face_model_name = "dmis-lab/biobert-v1.1"
        model_name_str = hugging_face_model_name.split("/")[1]
        annotated_data_path = bert_annotated_files_path_prefix + "ct_neuro_test_annotated_{}_{}.csv".format(model_name_str,
                                                                                                       "20240529")
        annotated_data_path_bio = bert_annotated_files_path_prefix + "ct_neuro_test_annotated_{}_BIO_{}.csv".format(
            model_name_str, "20240529")
        df = pd.read_csv(annotated_data_path)
        evaluator = ModelEvaluator(target_dataset, df, source_id_col_name="nct_id", source_sent_col_name='text',
                                   train_entities_distr_dict=train_entities_distr_dict,
                                   confidence=0.95, output_file_path=output_evaluation_path_prefix)
        print("*** Evaluating BIO format - ENTITY level***")

        print(evaluator.evaluate_bert_bio(annotated_data_path_bio, train_data_file,
                                          "ner_tags", return_format="all", target_labels_column="labels",
                                          predicted_labels_column=f"predictions_{model_name_str}"))
        print("*** Evaluating exact match format ***")
        print(evaluator.evaluate('ner_manual_ct_target', f'ner_prediction_{model_name_str}_normalized', model_name_str, ignore_labels_for_evaluation=False))

    if run_regex:
        print("Eval Regex model...")
        regex_annotations = "/Users/donevas/Desktop/Projects/Univeristy/PhD/Code/pythonNLP/clinical_trials_ner/corpus/second_annotation_round/model_annotations/ct_neuro_test_annotated_regex_20230922.csv"
        df = pd.read_csv(regex_annotations)
        evaluator = ModelEvaluator(target_dataset, df, source_id_col_name="nct_id", source_sent_col_name='text',
                                   train_entities_distr_dict=train_entities_distr_dict,
                                   confidence=0.95, output_file_path=output_evaluation_path_prefix)
        print(evaluator.evaluate_regex_bio(target_annotated_file_name=regex_annotations))
