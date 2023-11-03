import json
import csv
import random
from datetime import date
import pandas as pd
import math


def convert_text_to_json_official_random_text(row):
    k = random.randint(0, 1)  # randomly return 0 or 1
    if k == 0 and not pd.isnull(row['official_title']):
        return json.dumps({'nct_id': str(row['nct_id']), 'source': "OfficialTitle", 'text': row['official_title']})
    else:
        return json.dumps({'nct_id': str(row['nct_id']), 'source': "BriefSummary", 'text': row['description']})

# TODO: make more generic, currently very fixed for ClinicalTrials data!
def convert_text_to_json_official_combined_text(row):
    if not isinstance(row['official_title'], str) and (math.isnan(row['official_title'])):
        return json.dumps(
            {'nct_id': str(row['nct_id']), 'source': "OfficialTitle+BriefSummary", 'text': row['description']})
    else:
        return json.dumps({'nct_id': str(row['nct_id']), 'source': "OfficialTitle+BriefSummary",
                           'text': row['official_title'] + " | " + row['description']})


def save_text_json_to_csv(df, file_path_name):
    df['text_json'].to_csv(
        file_path_name, quoting=csv.QUOTE_NONE,
        header=None, index=None, sep='\n', mode='a')


if __name__ == '__main__':
    today = date.today()
    today_formatted = today.strftime("%Y%m%d")
    file_name_batch_1 = "data_aact_sample/random_sample_neurological_with_summaries_202306151741.csv"
    file_name_batch_2 = "data_aact_sample/aact_neuro_samples_second_batch_202309071141.csv"

    df1 = pd.read_csv(file_name_batch_1)

    # Load the second CSV file into another DataFrame
    df2 = pd.read_csv(file_name_batch_2)

    # Convert the 'nct_id' columns to sets and check for an overlap
    set1 = set(df1['nct_id'])
    set2 = set(df2['nct_id'])

    overlap = not set1.isdisjoint(set2)

    if overlap:
        print("The CSV files have an overlap in the 'nct_id' column.")
    else:
        print("The CSV files do not have an overlap in the 'nct_id' column.")

    df_clin_trials_sample = pd.read_csv(file_name_batch_2)
    df_clin_trials_sample['text_json'] = df_clin_trials_sample.apply(convert_text_to_json_official_combined_text,
                                                                     axis=1)
    save_text_json_to_csv(df_clin_trials_sample, "./data_for_prodigy"
                                  "/ct_random_neurological_{}_{}_second_batch.jsonl".format(500, today_formatted))

    process_first_batch = True
    # First annotations batch done in two-rounds, i.e. we first annotated 100 samples and refined the annotation guidelines, then another 400 samples were annotated
    if process_first_batch:
        df_clin_trials_sample = pd.read_csv(file_name_batch_1)
        df_clin_trials_sample['text_json'] = df_clin_trials_sample.apply(convert_text_to_json_official_combined_text,
                                                                         axis=1)
        df_100 = df_clin_trials_sample.head(100)  # First dataframe with 50 rows
        df_400 = df_clin_trials_sample.tail(400)

        save_text_json_to_csv(df_100, "./data_for_prodigy"
                                      "/ct_random_neurological_{}_{}.jsonl".format(100, today_formatted))
        save_text_json_to_csv(df_400,
                              "./data_for_prodigy"
                              "/ct_random_neurological_{}_{}.jsonl".format(400, today_formatted))
