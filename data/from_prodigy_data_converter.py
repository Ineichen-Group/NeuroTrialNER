import json
import pandas as pd


def extract_relevant_info_from_json_and_save_stats(input_file_path, output_file_path, file_name_addition):
    extracted_data = []
    labels_frequency = {}

    with open(input_file_path, "r") as file:
        count_lines = 0
        for line in file:
            count_lines += 1
            data = json.loads(line)

            nct_id = data["nct_id"]
            source = data["source"]
            text = data["text"]
            ner_manual = data.get("spans", [])

            parsed_annotations_all = [(ann['start'], ann['end'], ann['label'], text[ann['start']:ann['end']]) for
                                      # ann['end'] + 1 ??
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
                # TODO: not sure if I need to differentiate the column names
                "ner_manual_{}".format("ct_target"): parsed_annotations_all,
                "ner_manual_{}_idx".format("ct_target"): parsed_annotations_all_indices,
                "ner_manual_{}_disease".format("ct_target"): parsed_annotations_disease,
                "ner_manual_{}_intervention".format("ct_target"): parsed_annotations_intervention

            })
    print("Lines processed from jsonl: ", count_lines)

    df = pd.DataFrame(extracted_data)

    # Convert dictionary to DataFrame
    labels_frequency_df = pd.DataFrame(list(labels_frequency.items()), columns=['label', 'frequency'])

    # Save DataFrame to CSV file
    labels_frequency_df.to_csv(output_file_path + 'labels_frequency_{}_neuro.csv'.format(file_name_addition),
                               index=False)

    return df


# Prodigy outputs all version, i.e. for each annotator. Here we filter out only the resolved annotations for each element.
def remove_versions_from_resolved_dataset(file_prefix, input_file_name, output_file_name):
    with open(file_prefix + input_file_name + ".jsonl", 'r') as f, open(file_prefix + output_file_name + ".jsonl",
                                                                        'w') as output_file:
        for line in f:
            data = json.loads(line)
            if 'spans' in data:
                # clean the data from all the version of the different annotators
                output_file.write(
                    json.dumps({'nct_id': data['nct_id'], 'source': "OfficialTitle+BriefSummary", 'text': data['text'],
                                'tokens': data['tokens'], 'spans': data['spans'],
                                '_timestamp': data['_timestamp']}) + '\n')
            else:
                output_file.write(
                    json.dumps({'nct_id': data['nct_id'], 'source': "OfficialTitle+BriefSummary", 'text': data['text'],
                                'tokens': data['tokens'], 'spans': [],
                                '_timestamp': data['_timestamp']}) + '\n')


def filter_out_lines_for_review(file_name, annot_name,
                                file_with_nctids_to_filter="not_matching_nct_ids_annotations.csv"):
    file_prefix = "./annotated_data/final_combined/"
    df_not_matching = pd.read_csv(file_prefix + "{}".format(file_with_nctids_to_filter))
    nct_id_list_not_matching = df_not_matching["nct_id"].tolist()

    with open(file_prefix + file_name, 'r') as f, open(file_prefix + annot_name + "_filtered.jsonl",
                                                       'w') as output_file:
        for line in f:
            data = json.loads(line)
            nct_id = data['nct_id']
            if nct_id in nct_id_list_not_matching:
                output_file.write(json.dumps(data) + '\n')


def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = [json.loads(line.strip()) for line in lines]
    return data


def append_jsonl(file_path1, file_path2, output_file):
    data1 = read_jsonl(file_path1)
    data2 = read_jsonl(file_path2)

    combined_data = data1 + data2

    write_jsonl(output_file, combined_data)


def prepare_first_annotation_batch(file_prefix, output_stats_file_path="./annotated_data/corpus_stats/"):
    final_json_output_file_name = "ct_neuro_final_target_annotated_ds_round_1.jsonl"

    remove_versions_from_resolved_dataset(file_prefix, "neuro_merged_all_433", "neuro_merged_all_433_no_versions")
    remove_versions_from_resolved_dataset(file_prefix, "neuro_matching_annotations_reviewed_55",
                                          output_file_name="neuro_matching_annotations_reviewed_55_no_versions")
    append_jsonl(file_prefix + "neuro_merged_all_433_no_versions.jsonl",
                 file_prefix + "neuro_matching_annotations_reviewed_55_no_versions.jsonl",
                 output_file=file_prefix + final_json_output_file_name)

    df_annotations = extract_relevant_info_from_json_and_save_stats(
        file_prefix + final_json_output_file_name, output_stats_file_path,
        "batch1_433")
    df_annotations.to_csv(file_prefix + "ct_neuro_final_target_annotated_ds_round_1.csv")


def prepare_second_annotation_batch(file_prefix, output_stats_file_path="./annotated_data/corpus_stats/"):

    final_json_output_file_name = "ct_neuro_405_target_annotated_ds_round_2.jsonl"
    remove_versions_from_resolved_dataset(file_prefix, "neuro_merged_annotations_405_2batch",
                                          output_file_name="ct_neuro_405_target_annotated_ds_round_2")
    df_annotations = extract_relevant_info_from_json_and_save_stats(
        file_prefix + final_json_output_file_name,
        output_stats_file_path,
        "batch2_405")
    df_annotations.to_csv(file_prefix + "ct_neuro_final_target_annotated_ds_round_2.csv")


if __name__ == '__main__':
    prodigy_main_folder_path = "./"

    file_path_prefix_1 = prodigy_main_folder_path + "annotated_data" \
                                                    "/annotation_round_1/"
    file_path_prefix_2 = prodigy_main_folder_path + "annotated_data" \
                                                    "/annotation_round_2/"
    output_stats_file_path = prodigy_main_folder_path + "annotated_data/corpus_stats/"

    prepare_first_annotation_batch(file_path_prefix_1)
    prepare_second_annotation_batch(file_path_prefix_2)

    ### COMBINE THE DATA FROM THE TWO ANNOTATION ROUNDS
    json_output_file = prodigy_main_folder_path + "annotated_data/final_combined/" + "ct_neuro_final_target_annotated_ds_combined_rounds.jsonl"
    append_jsonl(file_path_prefix_1 + "ct_neuro_final_target_annotated_ds_round_1.jsonl",
                 file_path_prefix_2 + "ct_neuro_405_target_annotated_ds_round_2.jsonl",
                 output_file=json_output_file)
    df_annotations = extract_relevant_info_from_json_and_save_stats(
        json_output_file,
        prodigy_main_folder_path + "annotated_data/corpus_stats/",
        "combined_rounds")
    df_annotations.to_csv(
        prodigy_main_folder_path + "annotated_data/final_combined/ct_neuro_final_target_annotated_ds_combined_rounds.csv")

    # The below code was used when investigating differences between the annotators.
    # filter_out_lines_for_review("ben_annotated_500_neuro.jsonl", "matching", "matching_nct_ids_annotations.csv")
    # filter_out_lines_for_review("amelia_annotated_500_neuro.jsonl", "amelia")
    # filter_out_lines_for_review("simona_annotated_500_neuro.jsonl", "simona")
    # filter_out_lines_for_review("ben_annotated_500_neuro.jsonl", "ben")
