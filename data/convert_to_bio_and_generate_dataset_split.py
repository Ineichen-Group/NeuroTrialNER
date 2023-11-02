import json
import random
import pandas as pd


def read_json_split_data_to_csv(input_json_filename,
                                additional_annotation_format_file="ct_neuro_final_target_annotated_ds.csv",
                                output_data_path='corpus/test_annotations_for_guidlines/'):
    # Set a random seed for reproducibility
    random.seed(42)

    # Load the data from the file
    with open(input_json_filename, 'r') as file:
        data = [json.loads(row) for row in file]

    # Shuffle the data randomly
    random.shuffle(data)

    # Calculate the split indices
    total_samples = len(data)
    train_split = int(0.8 * total_samples)
    dev_split = int(0.9 * total_samples)

    # Split the data into train, dev, and test sets
    train_data = data[:train_split]
    dev_data = data[train_split:dev_split]
    test_data = data[dev_split:]

    # Check if dev and test sizes are not the same
    if len(dev_data) != len(test_data):
        # Remove the last row from the train set and append it to the smaller dataset
        removed_sample = train_data.pop()
        if len(dev_data) < len(test_data):
            dev_data.append(removed_sample)
        else:
            test_data.append(removed_sample)

    # Convert the data to DataFrame objects
    train_df = pd.DataFrame(train_data)
    dev_df = pd.DataFrame(dev_data)
    test_df = pd.DataFrame(test_data)

    # load the dataset that is not bio format
    additional_format = pd.read_csv(additional_annotation_format_file)[
        ["nct_id", "text", "ner_manual_ct_target"]]

    # Join and keep only the rows from each split
    train_merged = pd.merge(train_df, additional_format, left_on='id', right_on='nct_id', how='left')
    dev_merged = pd.merge(dev_df, additional_format, left_on='id', right_on='nct_id', how='left')
    test_merged = pd.merge(test_df, additional_format, left_on='id', right_on='nct_id', how='left')

    train_size = len(train_merged)
    dev_size = len(dev_merged)
    test_size = len(test_merged)

    # Save the merged DataFrames as CSV files
    train_merged.to_csv(output_data_path + 'ct_neuro_train_merged_{}.csv'.format(train_size), index=False)
    dev_merged.to_csv(output_data_path + 'ct_neuro_dev_merged_{}.csv'.format(dev_size), index=False)
    test_merged.to_csv(output_data_path + 'ct_neuro_test_merged_{}.csv'.format(test_size), index=False)

    # Save the splits into separate JSON files
    with open(output_data_path + 'ct_neuro_train_data_{}.json'.format(train_size), 'w') as file:
        for row in train_data:
            file.write(json.dumps(row) + '\n')

    with open(output_data_path + 'ct_neuro_dev_data_{}.json'.format(dev_size), 'w') as file:
        for row in dev_data:
            file.write(json.dumps(row) + '\n')

    with open(output_data_path + 'ct_neuro_test_data_{}.json'.format(test_size), 'w') as file:
        for row in test_data:
            file.write(json.dumps(row) + '\n')


def convert_annotations_to_bio_format(row):
    class_names = {
        "DRUG": "DRUG",
        "BEHAVIOURAL": "BEH",
        "SURGICAL": "SURG",
        "PHYSICAL": "PHYS",
        "RADIOTHERAPY": "RADIO",
        "OTHER": "OTHER",
        "CONDITION": "COND",
        "CONTROL": "CTRL"
    }

    tokens = [token["text"] for token in row["tokens"]]
    span_labels = ["O"] * len(tokens)

    if "spans" in row:
        for span in row["spans"]:
            token_start = span["token_start"]
            token_end = span["token_end"]
            label = span["label"]

            if token_start == token_end:
                span_labels[token_start] = f"B-{class_names.get(label)}"
            else:
                span_labels[token_start] = f"B-{class_names.get(label)}"
                span_labels[token_start + 1:token_end + 1] = [f"I-{class_names.get(label)}"] * (token_end - token_start)

    return {
        "tokens": tokens,
        "ner_tags": span_labels,
        "id": row["nct_id"]
    }


def process_file_bert_to_bio_format(input_jsonl_file_prodigy_format, output_jsonl_file_bio_format):
    print("Processing file: ", input_jsonl_file_prodigy_format)
    # Load the data from the file
    with open(input_jsonl_file_prodigy_format, 'r') as file:
        data = [json.loads(row) for row in file]

    # Convert each row into the new format
    converted_data = [convert_annotations_to_bio_format(row) for row in data]
    data_size = len(converted_data)

    # Save the converted data to a new file
    with open(output_jsonl_file_bio_format, 'w') as file:
        for row in converted_data:
            file.write(json.dumps(row) + '\n')
        # Calculate the maximum, minimum, and average number of tokens
        num_tokens = [len(row["tokens"]) for row in converted_data]
        max_tokens = max(num_tokens)
        min_tokens = min(num_tokens)
        avg_tokens = sum(num_tokens) / len(num_tokens)

        print("Maximum number of tokens:", max_tokens)
        print("Minimum number of tokens:", min_tokens)
        print("Average number of tokens:", avg_tokens)


if __name__ == '__main__':
    # used for the evaluation -> prodigy tokenizer is not the same as the BERT tokenizer -> when comparing the target annotation array from prodigy it is not the same to BERT
    # simulating how BERT will tokenize the sentence -> adjust the prodigy annotations to make it comparable
    files_directory = "./annotated_data/final_combined/"
    output_json_with_bio_format = files_directory + 'ct_neuro_final_target_annotated_ds_bio_format.jsonl'
    process_file_bert_to_bio_format(
        files_directory + '/ct_neuro_final_target_annotated_ds_combined_rounds.jsonl',
        output_jsonl_file_bio_format=output_json_with_bio_format)

    # SPLIT INTO TRAIN, DEV, TEST
    read_json_split_data_to_csv(output_json_with_bio_format,
                                additional_annotation_format_file=files_directory+"ct_neuro_final_target_annotated_ds_combined_rounds.csv",
                                output_data_path='./annotated_data/data_splits/')
