import spacy
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, Trainer
import pandas as pd
from multiprocessing import Pool
from .drugs_condition_dictionary_finder import find_drugs_normalized_output, \
    find_drugs_and_conditions_normalized_BIO_output
import re
import torch
from datasets import load_dataset
from .bert_helper import get_label_list, tokenize_and_align_labels
import numpy as np


class NERModel:
    def __init__(self, model_type, model_name, model_path=None, entity_class_names_dict=None):
        self.model_name = model_name
        if "/" in model_name:
            self.model_name_short = self.model_name.split("/")[1]
        else:
            self.model_name_short = self.model_name
        self.model_path = model_path
        self.model_type = model_type
        self.return_words_only = False
        self.entity_class_names_dict = entity_class_names_dict
        self.normalize_pred_representation = True  # used for the hugging face models - it removes the additional
        # information like prediction confidence that other models don't provide
        self.load_model()

    def load_model(self):
        if self.model_type == "spacy":
            self.nlp = spacy.load(self.model_path)
        elif self.model_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.config = AutoConfig.from_pretrained(self.model_path)
            self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer,
                                grouped_entities=True)  # grouped entities False to analyze the tokenization of the models
        elif self.model_type == "regex":
            pass
        else:
            raise ValueError("Wrong model type. Allowed values are spacy, regex, and huggingface.")

    def bert_predict_bio_format(self, ds_path_train, ds_path_test, text_column_name, label_column_name,
                                label_all_tokens=False, padding=True, max_length=512):
        model = self.model
        data_files = {"train": ds_path_train,
                      "test": ds_path_test}
        raw_datasets = load_dataset("json", data_files=data_files)
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
        id_to_label = {i: l for i, l in enumerate(label_list)}

        predict_dataset = raw_datasets["test"]
        predict_dataset = predict_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            desc="Running tokenizer on prediction dataset",
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "text_column_name": text_column_name,
                "label_column_name": label_column_name,
                "label_to_id": label_to_id,
                "label_all_tokens": label_all_tokens,
                "padding": padding,
                "max_length": max_length
            }
        )
        trainer = Trainer(
            model=model,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=self.tokenizer,
        )

        results = trainer.predict(predict_dataset)
        predictions = results.predictions
        predictions = np.argmax(predictions, axis=2)
        mapped_list_of_predictions = [np.array([id_to_label[id] for id in arr]) for arr in predictions]

        tokenized_inputs = self.tokenizer(
            predict_dataset['tokens'],
            padding=True,
            truncation=True,
            max_length=512,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        predict_dataset = predict_dataset.add_column("word_tokenized_input",
                                                     [tokenized_inputs.tokens(batch_index=i) for i in
                                                      range(len(predict_dataset))])
        predict_dataset_with_pred = predict_dataset.add_column(
            "predictions_bio_mapped_{}".format(self.model_name_short), mapped_list_of_predictions)
        predict_dataset_with_pred = predict_dataset_with_pred.add_column("predictions_{}".format(self.model_name_short),
                                                                         list(predictions))

        return predict_dataset_with_pred

    def annotate(self, file_path, source_column, sep=","):
        df = pd.read_csv(file_path, sep=sep)
        if self.normalize_pred_representation:
            predictions_col_name = "ner_prediction_{}_normalized".format(self.model_name_short)
        else:
            predictions_col_name = "ner_prediction_{}".format(self.model_name_short)
        # df[predictions_col_name] = df[source_column].apply(self.infer_ner)
        df[predictions_col_name] = df.apply(lambda row: self.infer_ner(row[source_column], row["tokens"]),
                                            axis=1)
        return df

    def annotate_parallel(self, file_path, source_column, sep=","):
        df = pd.read_csv(file_path, sep=sep)
        num_processes = 10
        chunks = [df[i:i + len(df) // num_processes] for i in
                  range(0, len(df), len(df) // num_processes)]

        with Pool(num_processes) as pool:
            # Apply the function to each chunk in parallel
            results = pool.starmap(self.infer_ner_chunk,
                                   [(chunk, source_column) for chunk in chunks])
        return pd.concat(results)

    def infer_ner_chunk(self, chunk, text_source_column):
        predictions_col_name = "ner_prediction_{}".format(self.model_name_short)
        chunk[predictions_col_name] = chunk[text_source_column].apply(self.infer_ner)
        if self.model_type == "huggingface":
            self.normalize_pred_representation = True
            predictions_col_name = "ner_prediction_{}_normalized".format(self.model_name_short)
            chunk[predictions_col_name] = chunk[text_source_column].apply(self.infer_ner)
            # predictions_col_name = "ner_prediction_{}_bio".format(self.model_type)
            # chunk[predictions_col_name] = chunk[text_source_column].apply(self.ner_bert_bio_output)
        return chunk

    def ner_bert_bio_output(self, sentence):
        # sentence = normalizer.normalize(sentence)

        model, tokenizer, labels = self.model, self.tokenizer, list(self.config.label2id.keys())
        #### check length limit ####
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        num_tokens = len(tokenized_sentence)
        if num_tokens > 512:
            tokenized_sentence = tokenized_sentence[:510]
            sentence = self.tokenizer.convert_tokens_to_string(tokenized_sentence)
        #### check length limit ####

        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
        inputs = tokenizer.encode(sentence, return_tensors="pt")
        outputs = model(inputs)[0]
        predictions = torch.argmax(outputs, axis=2)
        predictions = [(token, labels[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())]

        filtered_predictions = []
        prev_token, prev_label = None, None

        # dealing with the misaligned tokenizations
        # TODO: seems too manual... trying to fit to the the prodigy tokenization here
        for token, label in predictions:
            if prev_token == 'cann' and token == '##ot' or (
                    token == "##mg"):  # spacy splits connot into "can" and "not", two tokens
                filtered_predictions.append(label)
            elif token.startswith("##®") or (not token.startswith('##') and token not in ['[SEP]',
                                                                                          '[CLS]']):  # ignore the word pieces that would inflate the array
                filtered_predictions.append(label)
            prev_token, prev_label = token, label

        return filtered_predictions

    def infer_ner(self, sent, tokenized_sent=None):
        if self.model_type == "spacy":
            doc = self.nlp(sent)
            entities = []
            for ent in doc.ents:
                if self.return_words_only:  # TODO where is the right place?
                    entities.append(ent.text)
                else:
                    ent_label = ent.label_
                    if ent_label == "CHEMICAL":  # TODO: handle multi-class case!
                        entities.append((ent.start_char, ent.end_char, ent.text))
            return list(set(entities))
        elif self.model_type == "huggingface":
            tokenized_sentence = self.tokenizer.tokenize(sent)
            num_tokens = len(tokenized_sentence)
            if num_tokens > 512:
                print("Number of tokens ({}) too large, truncating sentence: {}".format(num_tokens, sent[:50] + "[...]"))
                tokenized_sentence = tokenized_sentence[:510]
                sent = self.tokenizer.convert_tokens_to_string(tokenized_sentence)
            ner_results = self.nlp(sent)
            # ner_results_combined = self.combine_entity_subwords(sent, ner_results) NOT NEEDED IF WE ARE USING THE GROUPED ENTITIES FLAG
            if self.normalize_pred_representation:
                return self.normalize_representation(ner_results)
            else:
                return ner_results
        elif self.model_type == "regex":
            tokens_cleaned = tokenized_sent
            return find_drugs_and_conditions_normalized_BIO_output(
                tokens_cleaned)  # returns a tuple drug_matches, {"entites":all_char_indices}
        else:
            raise ValueError("Wrong model type. Allowed values are spacy and huggingface.")

    def normalize_representation(self, ner_results_combined):
        results = []
        for ent_dict in ner_results_combined:
            if self.entity_class_names_dict:
                entity_class_full_name = self.entity_class_names_dict[ent_dict['entity_group']]
            else:
                entity_class_full_name = ent_dict['entity_group']
            results.append((ent_dict['start'], ent_dict['end'], entity_class_full_name, ent_dict['word']))
        return results


    def combine_entity_subwords(self, sent, entities):

        result = []
        current_entity = None
        merged_dict = {}
        if not entities:
            return []

        for entity in entities:
            if entity["entity"].startswith("B") or entity["entity"].startswith("I"):
                if current_entity is not None and entity["entity"].startswith("I"):
                    if entity['word'].startswith('##'):
                        current_entity['word'] += entity['word'][2:]
                    elif current_entity['end'] == entity['start']:
                        current_entity['word'] += entity['word']  # no space needed
                    else:
                        current_entity['word'] += " "
                        current_entity['word'] += entity['word']
                    current_entity['end'] = entity['end']
                    current_entity['score'] = round((current_entity['score'] + entity['score']) / 2,
                                                    3)  # average of the confidence
                else:
                    current_entity = entity.copy()
                    current_entity['entity'] = entity['entity'].replace("B-", "").replace("I-",
                                                                                          "")  # TODO: does the I- make sense?
                    if self.entity_class_names_dict:
                        current_entity['entity'] = self.entity_class_names_dict[current_entity['entity']]
                    current_entity['word'] = entity['word'][2:] if entity['word'].startswith('##') else entity['word']
                    result.append(current_entity)

        if self.return_words_only:
            current_entity_start = 0
            for i, entity in enumerate(result):

                if entity['entity'] == 'B':
                    merged_dict[i] = entity['word']
                    current_entity_start = i
                elif entity['entity'] == 'I':
                    if merged_dict:
                        merged_dict[current_entity_start] += ' ' + entity['word']
                    else:
                        print("could not save entity: {} from sentence {}. All entities found: {}".format(entity, sent,
                                                                                                          entities))

            return list(merged_dict.values())
        return result
