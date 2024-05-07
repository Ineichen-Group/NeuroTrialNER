import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import itertools


def extract_annotations(file_path):
    annotations = []

    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            spans = data.get("spans", [])

            annotations.append([span["label"] for span in spans])

    return annotations


def extract_relevant_info_from_json(file_path, annotator_name):
    extracted_data = []
    labels_frequency = {}

    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)

            nct_id = data["nct_id"]
            source = data["source"]
            text = data["text"]
            ner_manual = data.get("spans", [])

            parsed_annotations_all = [(ann['start'], ann['end'], ann['label'], text[ann['start']:ann['end'] + 1]) for
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
                "ner_manual_{}".format(annotator_name): parsed_annotations_all,
                "ner_manual_{}_idx".format(annotator_name): parsed_annotations_all_indices,
                "ner_manual_{}_disease".format(annotator_name): parsed_annotations_disease,
                "ner_manual_{}_intervention".format(annotator_name): parsed_annotations_intervention

            })

    df = pd.DataFrame(extracted_data)

    # Convert dictionary to DataFrame
    labels_frequency_df = pd.DataFrame(list(labels_frequency.items()), columns=['label', 'frequency'])

    # Save DataFrame to CSV file
    labels_frequency_df.to_csv(
        '/Users/donevas/Desktop/Projects/Univeristy/PhD/Code/pythonNLP/prodigy/annotated_data/labels_frequency_{}_neuro.csv'.format(
            annotator_name), index=False)

    return df


def extract_annotations_and_combine(annotated_files_list, annot1_name, annot2_name, annot3_name, sample_size):
    file_path_a1, file_path_a2, file_path_a3 = annotated_files_list

    df_1 = extract_relevant_info_from_json(file_path_a1, annot1_name)
    df_2 = extract_relevant_info_from_json(file_path_a2, annot2_name)
    if annot3_name != '':
        df_3 = extract_relevant_info_from_json(file_path_a3, annot3_name)

    merged_df = pd.merge(df_1, df_2[
        ["nct_id", "ner_manual_{}".format(annot2_name), "ner_manual_{}_idx".format(annot2_name),
         "ner_manual_{}_disease".format(annot2_name), "ner_manual_{}_intervention".format(annot2_name)]], on="nct_id",
                         how="inner")
    if annot3_name != '':
        merged_df = pd.merge(merged_df, df_3[
            ["nct_id", "ner_manual_{}".format(annot3_name), "ner_manual_{}_idx".format(annot3_name),
             "ner_manual_{}_disease".format(annot3_name), "ner_manual_{}_intervention".format(annot3_name)]],
                             on="nct_id",
                             how="inner")

    merged_df.to_csv(
        "/Users/donevas/Desktop/Projects/Univeristy/PhD/Code/pythonNLP/prodigy/annotated_data/annotated_{}_combined_neuro.csv".format(
            sample_size))


def filter_out_lines_for_review(file_name, annot_name,
                                file_with_nctids_to_filter="not_matching_nct_ids_annotations.csv"):
    file_prefix = "/Users/donevas/Desktop/Projects/Univeristy/PhD/Code/pythonNLP/prodigy/annotated_data/neuro_combined/"
    df_not_matching = pd.read_csv(
        "/Users/donevas/Desktop/Projects/Univeristy/PhD/Code/pythonNLP/prodigy/annotated_data/neuro_combined/{}".format(
            file_with_nctids_to_filter))
    nct_id_list_not_matching = df_not_matching["nct_id"].tolist()

    with open(file_prefix + file_name, 'r') as f, open(file_prefix + annot_name + "_filtered.jsonl",
                                                       'w') as output_file:
        for line in f:
            data = json.loads(line)
            nct_id = data['nct_id']
            if nct_id in nct_id_list_not_matching:
                output_file.write(json.dumps(data) + '\n')


def extract_annotated_array_df(file_path, annot_name):
    data_rows = []
    label_map = {
        "DRUG": 1,
        "BEHAVIOURAL": 2,
        "SURGICAL": 3,
        "PHYSICAL": 4,
        "RADIOTHERAPY": 5,
        "OTHER": 6,
        "CONDITION": 7,
        "CONTROL": 8
    }
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            nct_id = data['nct_id']
            sentence = data['text']
            spans = data.get("spans", [])

            annotations = ['0'] * len(data['tokens'])  # Initialize model_annotations with '0' for each token
            annotations_numeric = ['0'] * len(data['tokens'])  # Initialize model_annotations with '0' for each token

            for span in spans:
                label = span['label']
                token_start = span['token_start']
                token_end = span['token_end']

                for i in range(token_start, token_end + 1):
                    annotations[i] = label
                    annotations_numeric[i] = label_map[label]

            data_rows.append([nct_id, sentence, annotations, annotations_numeric])

    df = pd.DataFrame(data_rows, columns=['nct_id', 'Sentence', 'annotations_array_{}'.format(annot_name),
                                          'annotations_array_numeric_{}'.format(annot_name)])
    return df


def extract_annotations_and_combine_for_cohen_cappa(annotated_files_list, annot1_name, annot2_name, annot3_name='',
                                                    output_path_annotation_arrays=None):
    file_path_a1, file_path_a2, file_path_a3 = annotated_files_list

    df_1 = extract_annotated_array_df(file_path_a1, annot1_name)
    df_2 = extract_annotated_array_df(file_path_a2, annot2_name)
    if annot3_name != '':
        df_3 = extract_annotated_array_df(file_path_a3, annot3_name)

    merged_df = pd.merge(df_1, df_2[
        ["nct_id", 'annotations_array_{}'.format(annot2_name), 'annotations_array_numeric_{}'.format(annot2_name)]],
                         on="nct_id",
                         how="inner")
    if annot3_name != '':
        merged_df = pd.merge(merged_df, df_3[
            ["nct_id", 'annotations_array_{}'.format(annot3_name), 'annotations_array_numeric_{}'.format(annot3_name)]],
                             on="nct_id",
                             how="inner")

    merged_df.to_csv(output_path_annotation_arrays)


def calculate_overall_cohen_kappa_with_ci(df, annotators):
    # see implementation and explanation in https://rowannicholls.github.io/python/statistics/agreement/cohens_kappa.html

    for annotator1, annotator2 in itertools.combinations(annotators, 2):
        annotations1 = df[f'annotations_array_numeric_{annotator1}']  # ast.literal_eval(
        annotations2 = df[f'annotations_array_numeric_{annotator2}']

        # Combine all rows into a single array
        combined_array_1 = np.concatenate([eval(row) for row in annotations1]).tolist()
        combined_array_2 = np.concatenate([eval(row) for row in annotations2]).tolist()

        confusion = confusion_matrix(combined_array_1, combined_array_2)
        print(f"Cohen-Kappa with Confidence intervals {annotator1} vs {annotator2}")
        calculate_cohen_kappa_from_cfm_with_ci(confusion, print_result=True)


def calculate_overall_cohen_kappa(df, annotators):
    kappa_scores = []

    for annotator1, annotator2 in itertools.combinations(annotators, 2):
        annotations1 = df[f'annotations_array_numeric_{annotator1}']  # ast.literal_eval(
        annotations2 = df[f'annotations_array_numeric_{annotator2}']

        # Combine all rows into a single array
        combined_array_1 = np.concatenate([eval(row) for row in annotations1]).tolist()
        combined_array_2 = np.concatenate([eval(row) for row in annotations2]).tolist()

        kappa = cohen_kappa_score(combined_array_1, combined_array_2)
        kappa_scores.append((annotator1, annotator2, kappa))

    df_kappa_scores = pd.DataFrame(kappa_scores, columns=['Annotator 1', 'Annotator 2', 'Kappa Score'])
    print(df_kappa_scores)


def calculate_cohen_kappa_from_cfm_with_ci(confusion, print_result=False):
    # SOURCE FROM SKLEARN METRICS
    # Sample size
    n = np.sum(confusion)
    # Number of classes
    n_classes = confusion.shape[0]
    # Expected matrix
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    # SOURCE from: https://rowannicholls.github.io/python/statistics/agreement/cohens_kappa.html
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


def calculate_cohen_kappa_from_cfm(confusion):
    # SOURCE FROM SKLEARN METRICS
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    w_mat = np.ones([n_classes, n_classes], dtype=int)
    w_mat.flat[:: n_classes + 1] = 0

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k


def calculate_cohen_kappa_from_cfm_per_class(confusion, labels):
    n_classes = confusion.shape[0]

    # Calculate kappa score for each class
    kappa_per_class = []
    for i in range(1, n_classes):
        overlap = confusion[i, i]
        total_sum_all = np.sum(confusion)
        annot1 = np.sum(confusion[:, i]) - overlap
        annot2 = np.sum(confusion[i, :]) - overlap
        confusion_class = np.array([[overlap, annot1], [annot2, total_sum_all]])

        kappa, ci_boundary_limits = calculate_cohen_kappa_from_cfm_with_ci(confusion_class)
        lower = kappa - ci_boundary_limits
        upper = kappa + ci_boundary_limits
        print(
            f"Class {labels[i]}: lower: {round(lower, 3)}, {round(kappa, 3)} +/- {round(ci_boundary_limits, 3)}, upper: {round(upper, 3)}")
        kappa_per_class.append(kappa)

    return kappa_per_class


def vis_and_save_confusion_matrix(cm, labels, annotator1, annotator2, cf_output_path, suffix):
    # Divide values greater than 50,000 by 10
    cm[cm > 50000] = cm[cm > 50000] / 10

    # Create a ConfusionMatrixDisplay
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    cm_display.plot(cmap='Blues', values_format='.0f', xticks_rotation=90)
    # Set the fontsize for x and y axis labels
    cm_display.ax_.set_xlabel(annotator2, fontsize=12)  # Adjust the fontsize as needed
    cm_display.ax_.set_ylabel(annotator1, fontsize=12)

    # Adjust the font size of the labels
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    # Add a title
    # plt.title(f'Confusion matrix between {annotator1} and {annotator2}')

    # Adjust the layout parameters for better visibility
    plt.tight_layout()

    # Save the figure as a PNG image
    plt.savefig(cf_output_path + f'confusion_matrix_{annotator1}_{annotator2}_{suffix}.png')


def flatten_array_tuples(data):
    combined_array = []
    print(len(data[0]))
    for _, array_str in data:
        array = eval(array_str)
        combined_array.extend(array)
    return combined_array


def calculate_confusion_matrix_and_kappa_per_class(df, annotators, cf_output_path):
    labels = ['0', 'DRUG', 'BEHAVIOURAL', 'SURGICAL', 'PHYSICAL', 'RADIOTHERAPY', 'OTHER', 'CONDITION', 'CONTROL']
    annotator_abbreviations = {"simona": "SED", "benjamin": "BVI", "amelia": "AEC", "ben": "BVI"}

    for annotator1, annotator2 in itertools.combinations(annotators, 2):
        print("\n")
        print(f'Confusion matrix between {annotator1} and {annotator2}...')

        annotations1 = df[f'annotations_array_{annotator1}']  # ast.literal_eval(
        annotations2 = df[f'annotations_array_{annotator2}']

        combined_array_1 = np.concatenate([eval(row) for row in annotations1]).tolist()
        combined_array_2 = np.concatenate([eval(row) for row in annotations2]).tolist()

        cm = confusion_matrix(combined_array_1, combined_array_2, labels=labels)
        print("Kappa score per class: ")
        calculate_cohen_kappa_from_cfm_per_class(cm, labels)

        vis_and_save_confusion_matrix(cm, labels, annotator_abbreviations[annotator1],
                                      annotator_abbreviations[annotator2], cf_output_path, suffix="labels")


def convert_to_array(value_str):
    # Split the string and remove extra characters
    value_str = value_str.replace("['", "").replace("']", "").replace("'", "").split()
    # Convert the list of strings to a NumPy array
    return np.array(value_str)


if __name__ == '__main__':
    annotated_files_prefix_path = "./annotated_data"
    suffix = "_neuro"

    print("\n*** Combined Statistics Round 2 and 3 of annotations ***")
    file_path_a1 = "{}/annotation_round_3/bvi_neuro_ner_{}.jsonl".format(annotated_files_prefix_path,
                                                                         "round_2_3")
    file_path_a2 = "{}/annotation_round_3/sed_neuro_ner_{}.jsonl".format(annotated_files_prefix_path,
                                                                         "round_2_3")
    output_file = "{}/annotation_round_3/annotated_combined_arrays_neuro_round_2_3.csv".format(
        annotated_files_prefix_path)
    annotated_files_list = [file_path_a1, file_path_a2, '']
    extract_annotations_and_combine_for_cohen_cappa(annotated_files_list, "benjamin", "simona", '', output_file)
    df_round3 = pd.read_csv(output_file)
    calculate_overall_cohen_kappa(df_round3, ['simona', 'benjamin'])
    calculate_overall_cohen_kappa_with_ci(df_round3, ['simona', 'benjamin'])
    cf_output_path = "annotated_data/corpus_stats/annotations_confusion_matrix/round2_3_"
    calculate_confusion_matrix_and_kappa_per_class(df_round3, ['simona', 'benjamin'], cf_output_path=cf_output_path)

    print("\n*** Statistics Round 3 of annotations ***")
    file_path_a1 = "{}/annotation_round_3/bvi_neuro_ner_{}.jsonl".format(annotated_files_prefix_path,
                                                                         "non_drug")
    file_path_a2 = "{}/annotation_round_3/sed_neuro_ner_{}.jsonl".format(annotated_files_prefix_path,
                                                                         "non_drug")
    output_file = "{}/annotation_round_3/annotated_combined_arrays_neuro_round_3.csv".format(
        annotated_files_prefix_path)
    annotated_files_list = [file_path_a1, file_path_a2, '']
    extract_annotations_and_combine_for_cohen_cappa(annotated_files_list, "benjamin", "simona", '', output_file)
    df_round3 = pd.read_csv(output_file)
    calculate_overall_cohen_kappa(df_round3, ['simona', 'benjamin'])
    calculate_overall_cohen_kappa_with_ci(df_round3, ['simona', 'benjamin'])
    cf_output_path = "annotated_data/corpus_stats/annotations_confusion_matrix/round3_"
    calculate_confusion_matrix_and_kappa_per_class(df_round3, ['simona', 'benjamin'], cf_output_path=cf_output_path)

    print("\n*** Statistics Round 2 of annotations ***")
    file_path_a1 = "{}/annotation_round_2/benjamin_ct_ds_500_2batch{}.jsonl".format(annotated_files_prefix_path,
                                                                                    suffix)
    file_path_a2 = "{}/annotation_round_2/simona_ct_ds_500_2batch{}.jsonl".format(annotated_files_prefix_path,
                                                                                  suffix)
    output_file = "{}/annotation_round_2/annotated_combined_arrays_neuro_round2.csv".format(annotated_files_prefix_path)
    annotated_files_list = [file_path_a1, file_path_a2, '']
    extract_annotations_and_combine_for_cohen_cappa(annotated_files_list, "benjamin", "simona", '', output_file)
    df_round2 = pd.read_csv(output_file)
    calculate_overall_cohen_kappa(df_round2, ['simona', 'benjamin'])
    calculate_overall_cohen_kappa_with_ci(df_round2, ['simona', 'benjamin'])
    cf_output_path = "annotated_data/corpus_stats/annotations_confusion_matrix/round2_"
    calculate_confusion_matrix_and_kappa_per_class(df_round2, ['simona', 'benjamin'], cf_output_path=cf_output_path)

    print("\n*** Statistics Round 1 of annotations ***")
    sample_size = 500
    file_path_a1 = "{}/annotation_round_1/amelia_annotated_{}{}.jsonl".format(annotated_files_prefix_path, sample_size,
                                                                              suffix)
    file_path_a2 = "{}/annotation_round_1/ben_annotated_{}{}.jsonl".format(annotated_files_prefix_path, sample_size,
                                                                           suffix)
    file_path_a3 = "{}/annotation_round_1/simona_annotated_{}{}.jsonl".format(annotated_files_prefix_path, sample_size,
                                                                              suffix)
    annotated_files_list = [file_path_a1, file_path_a2, file_path_a3]

    output_file = "annotated_data/annotation_round_1/annotated_{}_combined_arrays_neuro_round1.csv".format(sample_size)

    extract_annotations_and_combine_for_cohen_cappa(annotated_files_list, "amelia", "ben", "simona", output_file)

    df_round1 = pd.read_csv(output_file)
    calculate_overall_cohen_kappa_with_ci(df_round1, ['amelia', 'simona', 'ben'])
    cf_output_path = "annotated_data/corpus_stats/annotations_confusion_matrix/round1_"
    calculate_confusion_matrix_and_kappa_per_class(df_round1, ['amelia', 'simona', 'ben'],
                                                   cf_output_path=cf_output_path)
