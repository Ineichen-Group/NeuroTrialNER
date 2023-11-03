import pandas as pd
import re
from collections import Counter
import string
import numpy as np
from scipy.stats import t
import json
from collections import defaultdict


def extract_relevant_info_from_json(file_path, file_name_addition, output_path):
    extracted_data = []
    labels_frequency = {}

    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)

            nct_id = data["nct_id"]
            source = data["source"]
            text = data["text"]
            ner_manual = data.get("spans", [])

            parsed_annotations_all = [(ann['start'], ann['end'], ann['label'], text[ann['start']:ann['end']]) for
                                      ann in ner_manual]
            parsed_annotations_all_indices = [(ann['start'], ann['end']) for ann in ner_manual]
            parsed_annotations_disease = [(ann['start'], ann['end'], ann['label']) for ann in ner_manual if
                                          ann['label'] == 'CONDITION']
            parsed_annotations_intervention = [(ann['start'], ann['end'], ann['label']) for ann in ner_manual if
                                               ann['label'] != 'CONDITION']

            # Collect label frequencies
            for ann in ner_manual:
                label = ann['label']
                if label in labels_frequency:
                    labels_frequency[label] += 1
                else:
                    labels_frequency[label] = 1

            extracted_data.append({
                "nct_id": nct_id,
                "source": source,
                "text": text,
                "ner_manual_{}".format(file_name_addition): parsed_annotations_all,
                "ner_manual_{}_idx".format(file_name_addition): parsed_annotations_all_indices,
                "ner_manual_{}_disease".format(file_name_addition): parsed_annotations_disease,
                "ner_manual_{}_intervention".format(file_name_addition): parsed_annotations_intervention

            })

    df = pd.DataFrame(extracted_data)

    # Convert dictionary to DataFrame
    labels_frequency_df = pd.DataFrame(list(labels_frequency.items()), columns=['label', 'frequency'])

    # Save DataFrame to CSV file
    labels_frequency_df.to_csv(output_path + 'labels_frequency_{}_neuro.csv'.format(
        file_name_addition), index=False)

    return df


def collect_ds_statistics(df, ds_name, ds_split, column_with_annotations='Target_NER', output_path='experiments/'):
    statistics = defaultdict(lambda: {'frequency': 0, 'entity_class': ''})
    entity_class_totals = defaultdict(int)  # To store the aggregated frequencies per entity_class

    # Parse the dataframe and calculate the statistics
    for index, row in df.iterrows():
        target_ner = row[column_with_annotations]
        ner_list = eval(target_ner)

        # Process each entity in the list
        for entity in ner_list:
            start, end, entity_class, entity_token = entity
            entity_token_lower = entity_token.lower()

            # Update the statistics
            statistics[entity_token_lower]['frequency'] += 1
            statistics[entity_token_lower]['entity_class'] = entity_class

            # Update the aggregated frequency for the entity_class
            entity_class_totals[entity_class] += 1

    # Convert the statistics to the desired format, sort it by frequency
    result = [{'entity_token': token, **data} for token, data in statistics.items()]
    result = sorted(result, key=lambda x: x['frequency'], reverse=True)
    result_df = pd.DataFrame(result)
    result_df.to_csv(output_path + '{}_{}_entities_stats.csv'.format(ds_name, ds_split), index=False)

    # Create a separate file for aggregated frequencies per entity_class and sort it by frequency
    entity_class_result = [{'entity_class': ec, 'frequency': freq} for ec, freq in entity_class_totals.items()]
    entity_class_result = sorted(entity_class_result, key=lambda x: x['frequency'], reverse=True)
    entity_class_df = pd.DataFrame(entity_class_result)
    entity_class_df.to_csv(output_path + '{}_{}_entity_class_stats.csv'.format(ds_name, ds_split), index=False)


def calculate_average_words_and_sentences_confidence(csv_file_path, confidence_level=0.95):
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Calculate the average number of words and sentences in the text
    total_words = 0
    total_sentences = 0
    num_texts = len(data)
    for text in data['text']:
        # Count words
        words = re.findall(r'\w+', text)
        total_words += len(words)

        # Count sentences
        sentences = re.split(r'[.!?]', text)
        total_sentences += len(sentences)

    average_words = total_words / num_texts
    average_sentences = total_sentences / num_texts

    # Calculate the standard deviation for words
    word_counts = [len(re.findall(r'\w+', text)) for text in data['text']]
    std_dev_words = np.std(word_counts, ddof=1)

    # Calculate the standard deviation for sentences
    sentence_counts = [len(re.split(r'[.!?]', text)) for text in data['text']]
    std_dev_sentences = np.std(sentence_counts, ddof=1)

    # Calculate the confidence interval for words
    alpha = 1 - confidence_level
    t_critical = t.ppf(1 - alpha / 2, num_texts - 1)
    margin_of_error_words = t_critical * std_dev_words / np.sqrt(num_texts)
    lower_bound_words = average_words - margin_of_error_words
    upper_bound_words = average_words + margin_of_error_words

    # Calculate the confidence interval for sentences
    margin_of_error_sentences = t_critical * std_dev_sentences / np.sqrt(num_texts)
    lower_bound_sentences = average_sentences - margin_of_error_sentences
    upper_bound_sentences = average_sentences + margin_of_error_sentences

    # Return the average number of words, sentences, and confidence intervals
    return {
        'average_words': round(average_words, 2),
        'average_sentences': round(average_sentences, 2),
        'confidence_interval_words': (round(lower_bound_words, 2), round(upper_bound_words, 2)),
        'confidence_interval_sentences': (round(lower_bound_sentences, 2), round(upper_bound_sentences, 2)),
        'std_dev_words': margin_of_error_words,
        'std_dev_sentences': margin_of_error_sentences
    }


if __name__ == '__main__':
    train_file = 'annotated_data/data_splits/ct_neuro_train_merged_713.csv'
    dev_file = 'annotated_data/data_splits/ct_neuro_dev_merged_90.csv'
    test_file = 'annotated_data/data_splits/ct_neuro_test_merged_90.csv'

    print(calculate_average_words_and_sentences_confidence(train_file))
    print(calculate_average_words_and_sentences_confidence(dev_file))
    print(calculate_average_words_and_sentences_confidence(test_file))

    collect_ds_statistics(pd.read_csv(train_file), ds_name="clintrials", ds_split="train",
                          column_with_annotations='ner_manual_ct_target', output_path='./annotated_data/corpus_stats/')
    collect_ds_statistics(pd.read_csv(dev_file), ds_name="clintrials", ds_split="dev",
                          column_with_annotations='ner_manual_ct_target', output_path='./annotated_data/corpus_stats/')
    collect_ds_statistics(pd.read_csv(test_file), ds_name="clintrials", ds_split="test",
                          column_with_annotations='ner_manual_ct_target', output_path='./annotated_data/corpus_stats/')
