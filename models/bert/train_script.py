import os
import argparse
import wandb
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def tokenize_and_align_labels(examples, tokenizer, text_column_name, label_column_name, label_to_id,
                              label_all_tokens=False, padding=True, max_length=512):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length=max_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    word_ids_list = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_ids_list.append(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_to_id[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_ids"] = word_ids_list
    return tokenized_inputs


# Define your function for computing metrics here
def compute_metrics(p, label_list):
    metric = load_metric("seqeval")
    return_entity_level_metrics = False
    return_macro_metrics = False

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # This is just flattening the result dict
        # e.g. {'MISC': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'number': 1}, 'PER': {'precision': 1.0, 'recall': 0.5, 'f1': 0.66, 'number': 2}, 'overall_precision': 0.5, 'overall_recall': 0.33, 'overall_f1': 0.4, 'overall_accuracy': 0.66}
        # -> {'MISC_precision': 0.0, 'MISC_recall': 0.0, 'MISC_f1': 0.0, 'MISC_number': 1, 'PER_precision': 1.0, 'PER_recall': 0.5, 'PER_f1': 0.66, 'PER_number': 2, 'overall_precision': 0.5, 'overall_recall': 0.33, 'overall_f1': 0.4, 'overall_accuracy': 0.66}
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    if return_macro_metrics:
        Ps, Rs, Fs = [], [], []
        for type_name in results:
            if type_name.startswith("overall"):
                continue
            print('type_name', type_name)
            Ps.append(results[type_name]["precision"])
            Rs.append(results[type_name]["recall"])
            Fs.append(results[type_name]["f1"])
        return {
            "macro_precision": np.mean(Ps),
            "macro_recall": np.mean(Rs),
            "macro_f1": np.mean(Fs),
        }
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Train and evaluate model.")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--percentage", type=int, default=100, help="Percentage value")
    parser.add_argument("--i", type=int, default=1, help="Iteration value")
    parser.add_argument("--train_data_path", default=None, type=str, help="Path to the train data file")
    parser.add_argument("--val_data_path", default=None, type=str, help="Path to the validation data file")
    parser.add_argument("--test_data_path", default=None, type=str, help="Path to the test data file")
    parser.add_argument("--output_path", default=None, type=str, help="Path for the output.")
    parser.add_argument("--model_name_or_path", default='michiyasunaga/BioLinkBERT-base', type=str,
                        help="HuggingFace Model.")

    args = parser.parse_args()
    n_epochs = args.n_epochs
    percentage = args.percentage
    i = args.i
    ds_path_train = args.train_data_path
    ds_path_test = args.test_data_path
    ds_path_val = args.val_data_path
    model_name_or_path = args.model_name_or_path
    model_name_str = model_name_or_path.replace("/", "_")
    output_folder_path = args.output_path

    ## DATA LOAD AND PREPARATION
    logger.info("*** PREPARE DATA ***")
    logger.info(f"Train dataset path: {ds_path_train}")
    text_column_name = "tokens"
    label_column_name = "ner_tags"
    label_all_tokens = False
    padding = True
    max_length = 512
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir="data/huggingface_cache")

    data_files = {"train": ds_path_train,
                  "validation": ds_path_val,
                  "test": ds_path_test}

    raw_datasets = load_dataset("json", data_files=data_files)
    label_list = get_label_list(raw_datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}

    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_column_name": text_column_name,
            "label_column_name": label_column_name,
            "label_to_id": label_to_id,
            "label_all_tokens": label_all_tokens,
            "padding": padding,
            "max_length": max_length
        }
    )

    valid_dataset = raw_datasets["validation"]
    valid_dataset = valid_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Running tokenizer on validation dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_column_name": text_column_name,
            "label_column_name": label_column_name,
            "label_to_id": label_to_id,
            "label_all_tokens": label_all_tokens,
            "padding": padding,
            "max_length": max_length
        }
    )

    predict_dataset = raw_datasets["test"]
    predict_dataset = predict_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Running tokenizer on prediction dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_column_name": text_column_name,
            "label_column_name": label_column_name,
            "label_to_id": label_to_id,
            "label_all_tokens": label_all_tokens,
            "padding": padding,
            "max_length": max_length
        }
    )

    suffix = f"epochs_{n_epochs}_data_size_{percentage}_iter_{i}"
    run_name = "{}_{}".format(model_name_or_path, suffix)  # Different run_name for each iteration
    num_labels = len(label_list)

    logger.info("*** CONFIGURE AND LOAD MODEL ***")

    # Configure your model's architecture
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        cache_dir="data/huggingface_cache"
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir="data/huggingface_cache"
    )

    # SAMPLE TRAIN SET
    logger.info("*** Sample from the training set based on the given percentage. ***")
    # Calculate the number of samples based on the percentage
    num_samples = int(len(train_dataset) * percentage / 100)
    # Reseed the random number generator
    np.random.seed()  # This will use a different seed every time
    # Sample the dataset based on the calculated number of samples
    sampled_indices = np.random.choice(len(train_dataset), size=num_samples, replace=False)
    sampled_subset = train_dataset.select(sampled_indices)
    print("Training with train size: ", len(sampled_subset))

    logger.info("*** CONFIGURE AND LOAD TRAINER ***")

    training_args = TrainingArguments(
        output_dir=output_folder_path + '/results/' + model_name_str + "/" + suffix,  # output directory
        num_train_epochs=n_epochs,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        # warmup_steps=500,                # number of warmup steps for learning rate scheduler
        warmup_ratio=0.1,
        weight_decay=0.01,  # strength of weight decay
        logging_dir=output_folder_path + '/logs/' + suffix,  # directory for storing logs
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        # limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`. When `load_best_model_at_end` is enabled, the "best" checkpoint will always be retained in addition to the most recent ones.
        report_to="wandb",
        run_name=run_name)

    # Initialize WandB
    wandb.init(
        project="clinical-trials-ner",
        group="train-data-size-impact-cluster",
        name=run_name,
        config=training_args
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=sampled_subset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, label_list)
    )
    ### TRAIN
    logger.info("*** TRAIN ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.log(metrics)
    trainer.save_model()  # will save the best model since load_best_model_at_end=True

    ### EVAL
    logger.info("*** EVALUATE ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(valid_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    ### TEST
    logger.info("*** TEST ***")
    results = trainer.predict(predict_dataset, metric_key_prefix="test")
    predictions = results.predictions
    metrics = results.metrics
    metrics["test_samples"] = len(predict_dataset)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    trainer.log(metrics)

    # Close the WandB session
    wandb.finish()


if __name__ == "__main__":
    main()
