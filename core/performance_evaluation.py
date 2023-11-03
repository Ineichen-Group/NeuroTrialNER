import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from scipy.stats import norm
from collections import defaultdict
from .bert_helper import get_label_list, format_output_seqeval
from datasets import load_dataset, load_metric
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


class ModelEvaluator:

    def __init__(self, dataset_name=None, annotated_target_df=None, source_id_col_name=None, source_sent_col_name=None,
                 train_entities_distr_dict=None, confidence=0.95, output_file_path="experiments/"):
        self.ds_name = dataset_name
        self.source = source_sent_col_name
        self.col_id = source_id_col_name
        self.df = annotated_target_df
        self.confidence = confidence
        self.train_entities_distribution = train_entities_distr_dict
        self.output_file_path = output_file_path

    def evaluate_confusion_matrix(self, confusion_matrix):
        tp, fp, fn, tn = confusion_matrix
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        # calculate F1 score
        f1 = self.f1_score(precision, recall)

        # calculate Wilson score interval
        interval_precision = self.wilson_score_interval("precision", confusion_matrix)
        interval_recall = self.wilson_score_interval("recall", confusion_matrix)

        # return results as a dictionary
        results = {
            'confusion_matrix': {
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TP': tp
            },
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'wilson_score_interval_p': interval_precision,
            'wilson_score_interval_r': interval_recall
        }
        return results

    def calculate_cohen_kappa_from_cfm_with_ci(self, confusion, print_result=False):
        # COPIED FROM SKLEARN METRICS
        # Sample size
        n = np.sum(confusion)
        # Number of classes
        n_classes = confusion.shape[0]
        # Expected matrix
        sum0 = np.sum(confusion, axis=0)
        sum1 = np.sum(confusion, axis=1)
        expected = np.outer(sum0, sum1) / np.sum(sum0)

        # Calculate p_o (the observed proportionate agreement) and
        # p_e (the probability of random agreement)
        identity = np.identity(n_classes)
        p_o = np.sum((identity * confusion) / n)
        p_e = np.sum((identity * expected) / n)
        # Calculate Cohen's kappa
        kappa = (p_o - p_e) / (1 - p_e)
        # Confidence intervals
        se = np.sqrt((p_o * (1 - p_o)) / (n * (1 - p_e) ** 2))
        ci = 1.96 * se * 2
        ci_boundary_limits = 1.96 * se
        lower = kappa - ci_boundary_limits
        upper = kappa + ci_boundary_limits

        if print_result:
            print(
                f'p_o = {p_o}, p_e = {p_e}, lower={lower:.2f}, kappa = {kappa:.2f}, upper={upper:.2f}, boundary = {ci_boundary_limits:.3f}\n',
                f'standard error = {se:.3f}\n',
                f'lower confidence interval = {lower:.3f}\n',
                f'upper confidence interval = {upper:.3f}', sep=''
            )

        return kappa, ci_boundary_limits
    def calculate_overall_cohen_kappa_with_ci(self, annotations1, annotations2):
        # see implementation and explanation in https://rowannicholls.github.io/python/statistics/agreement/cohens_kappa.html

        confusion = confusion_matrix(annotations1, annotations2)
        print("Cohen-Kappa with Confidence intervals Model vs Target")
        self.calculate_cohen_kappa_from_cfm_with_ci(confusion, print_result=True)

    def evaluate_bert_bio(self, target_annotated_file_name, train_file_name, train_label_column_name,
                          return_format="all", target_labels_column="labels", predicted_labels_column="predictions"):
        df = pd.read_csv(target_annotated_file_name)

        def convert_to_list(string):
            string = string.strip('[]')  # Remove the brackets
            return list(map(int, string.split()))

        predictions = np.array(df[predicted_labels_column].apply(convert_to_list).to_list())
        labels = np.array(df[target_labels_column].apply(convert_to_list).to_list())

        data_files = {"train": train_file_name}
        raw_datasets = load_dataset("json", data_files=data_files)
        label_list = get_label_list(raw_datasets["train"][train_label_column_name])

        print("predicted_labels_column: ", predicted_labels_column)
        print("len: ", len(predictions))
        print("target_labels_column: ", target_labels_column)
        print("len: ", len(labels))

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        metric = load_metric("seqeval")
        results = metric.compute(predictions=true_predictions, references=true_labels)

        combined_predictions = [item for sublist in true_predictions for item in sublist]
        combined_target = [item for sublist in true_labels for item in sublist]
        self.calculate_overall_cohen_kappa_with_ci(combined_predictions, combined_target)

        print(classification_report(true_labels, true_predictions))
        print("Mode STRICT")
        print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))

        return format_output_seqeval(results, return_format)

    def evaluate_regex_bio(self, target_annotated_file_name,
                          return_format="all", target_labels_column="ner_tags", predicted_labels_column="ner_prediction_regex_normalized"):
        df = pd.read_csv(target_annotated_file_name)

        def convert_to_list(string):
            cleaned_string = string.replace("'", "").replace(",", "").strip('[]')
            return cleaned_string.split()

        # Assuming df is your DataFrame and you want to convert columns to lists of strings
        predicted_labels_column = predicted_labels_column
        target_labels_column = target_labels_column

        # Apply the convert_to_list function to each element of the Series
        predictions = df[predicted_labels_column].apply(convert_to_list)
        labels = df[target_labels_column].apply(convert_to_list)
        predictions = predictions.to_numpy()
        labels = labels.to_numpy()

        metric = load_metric("seqeval")
        results = metric.compute(predictions=predictions, references=labels)

        print(classification_report(labels, predictions))
        print("Mode STRICT")
        print(classification_report(labels, predictions, mode='strict', scheme=IOB2))

        return format_output_seqeval(results, return_format)

    def evaluate(self, actual_col_name, predicted_col_name, model_name, ignore_labels_for_evaluation=False):
        self.actual = actual_col_name
        self.predicted = predicted_col_name
        self.model_name = model_name
        self.ignore_labels_for_evaluation = ignore_labels_for_evaluation

        results = self.evaluate_confusion_matrix(self.calculate_confusion_matrix())

        return results

    def update_entity_error_analysis_stats(self, col_id, stats, stats_sent, entity_set, eval_type, target_entities,
                                           prediction_set):

        entity_class = "tbd"  # default class #TODO: works only for the BC5CDR-drug case, generalize!
        for entity in entity_set:
            entity_details = eval(entity)
            if len(entity_details) == 3:
                start, end, entity_token = eval(entity)
            else:
                start, end, entity_class, entity_token = eval(entity)
            entity_token_lower = entity_token.lower()
            # Update the statistics
            stats[entity_token_lower]['in_test_frq'] += 1
            stats[entity_token_lower]['entity_class'] = entity_class
            stats[entity_token_lower]['evaluation'][eval_type] = stats[entity_token_lower]['evaluation'][eval_type] + 1
            # stats[entity_token_lower]['sentences'][eval_type].append(sent)
            if self.train_entities_distribution.get(entity_token_lower):
                stats[entity_token_lower]['in_train_frq'] = self.train_entities_distribution[entity_token_lower][
                    'frequency']
            else:
                stats[entity_token_lower]['in_train_frq'] = 0

            # stats_sent[entity_token_lower]['sentences'].append(sent)
            # stats_sent[entity_token_lower]['evaluation'].append(eval_type)
            local_df = pd.DataFrame(
                {'entity_token': [entity_token_lower], 'nct_id': [col_id], 'entity_class': [entity_class],
                 'evaluation': [eval_type],
                 'target': [target_entities], 'predicted': [prediction_set]})
            stats_sent = pd.concat([stats_sent, local_df])

        return stats, stats_sent

    def calculate_confusion_matrix(self):
        tp, fp, fn, tn = 0, 0, 0, 0
        df = self.df
        statistics = defaultdict(lambda: {'in_test_frq': 0, 'entity_class': '', 'in_train_frq': 0,
                                          'evaluation': {'tp': 0, 'fp': 0, 'fn': 0}})
        df_sent_level_stats = pd.DataFrame(columns=['entity_token', 'nct_id', 'evaluation', 'target'])

        for _, row in df.iterrows():
            col_id = row[self.col_id]
            target = eval(row[self.actual])
            prediction = eval(row[self.predicted])

            if self.ignore_labels_for_evaluation:
                target = [(t[0], t[1], t[3]) for t in target]
                prediction = [(t[0], t[1], t[3]) for t in prediction]

            # convert list of tuples to set of strings for easier comparison
            target_set = set([str(t).lower() for t in target])
            prediction_set = set([str(p).lower() for p in prediction])

            # calculate true positives
            tp_set = target_set.intersection(prediction_set)
            statistics, df_sent_level_stats = self.update_entity_error_analysis_stats(col_id, statistics,
                                                                                      df_sent_level_stats, tp_set, "tp",
                                                                                      target_set, prediction_set)
            tp += len(tp_set)

            # calculate false positives
            fp_set = prediction_set - target_set
            statistics, df_sent_level_stats = self.update_entity_error_analysis_stats(col_id, statistics,
                                                                                      df_sent_level_stats, fp_set, "fp",
                                                                                      target_set, prediction_set)
            fp += len(fp_set)

            # calculate false negatives
            fn_set = target_set - prediction_set
            statistics, df_sent_level_stats = self.update_entity_error_analysis_stats(col_id, statistics,
                                                                                      df_sent_level_stats, fn_set, "fn",
                                                                                      target_set, prediction_set)
            fn += len(fn_set)

            # calculate true negatives
            source_sentence_tokens = row[self.source].split(" ")
            tn += len(source_sentence_tokens) - len(target_set.union(prediction_set))

        stats_result = [{'entity_token': token, **data} for token, data in statistics.items()]
        stats_result_df = pd.DataFrame(stats_result)
        stats_result_df['model'] = self.model_name
        stats_result_df.to_csv(
            self.output_file_path + '{}_{}_entities_stats_{}.csv'.format(self.ds_name, "evaluation", self.model_name),
            index=False)

        # stats_result = [{'entity_token': token, **outputs} for token, outputs in statistics_sent.items()]
        # stats_result_df = pd.DataFrame(stats_result)
        df_sent_level_stats['model'] = self.model_name
        df_sent_level_stats.to_csv(
            self.output_file_path + '{}_{}_sent_entities_stats_{}.csv'.format(self.ds_name, "evaluation",
                                                                              self.model_name),
            index=False)

        return tp, fp, fn, tn

    def f1_score(self, precision, recall):
        if precision > 0 or recall > 0:
            f1 = 2 * ((precision * recall) / (precision + recall))
        else:
            f1 = 0
        return f1

    # Implementation reference: Held, Leonhard, and Daniel Sabanés Bové. "Likelihood and Bayesian Inference."
    # Statistics for Biology and Health. Springer, Berlin, Heidelberg, (2020). p. 115
    def wilson_score_interval(self, metric, confusion_matrix=None):
        if not confusion_matrix:
            tp, fp, fn, tn = self.calculate_confusion_matrix()
        else:
            tp, fp, fn, tn = confusion_matrix
        x = tp
        if metric == "precision":
            n = tp + fp
        elif metric == "recall":
            n = tp + fn
        else:
            return (0, 0)

        z = norm.ppf(1 - (1 - self.confidence) / 2)
        phat = x / n
        center = (x + z ** 2 / 2) / (n + z ** 2)

        interval = ((z * np.sqrt(n)) / (n + z ** 2)) * np.sqrt(phat * (1 - phat) + z ** 2 / (4 * n))

        lower_bound = center - interval
        upper_bound = center + interval
        return lower_bound, phat, upper_bound
