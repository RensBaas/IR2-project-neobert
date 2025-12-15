from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DebertaV2Tokenizer
import torch.nn as nn
from evaluate import load
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
)
import argparse

label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def combine_labels(data):
    data["labels"] = [float(data[col]) for col in label_columns]
    return data

# {'id': '02141412314', 'comment_text': 'Sample comment text', 'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 1, }
def get_test_dataset():
    dataset = load_dataset("csv", data_files="test.csv")["train"]
    labels = load_dataset("csv", data_files="test_labels.csv")["train"]
    labels = labels.map(combine_labels)

    dataset = dataset.add_column("labels", labels["labels"])
    return dataset

def get_tokenizer(tokenizer_name):
    if tokenizer_name == "microsoft/deberta-v3-large":      
        tokenizer = DebertaV2Tokenizer.from_pretrained(
                "microsoft/deberta-v3-large",
                use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return tokenizer

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["comment_text"], padding="max_length", truncation=True, max_length=512)


def get_model(model_location):
    model = AutoModelForSequenceClassification.from_pretrained(model_location, trust_remote_code=True, num_labels=6, problem_type="multi_label_classification")

    return model


# Define a custom compute_metrics function\
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))   # sigmoid on numpy array
    predictions = (probs > 0.5).astype(int)
    # Mask unknown labels
    mask = labels != -1
    labels_masked = labels[mask]
    preds_masked  = predictions[mask]
    probs_masked  = probs[mask]

    return {
        "f1_micro": f1_score(labels_masked, preds_masked, average="micro"),
        "f1_macro": f1_score(labels_masked, preds_masked, average="macro"),
        "f1_weighthed": f1_score(labels_masked, preds_masked, average="weighted"),
        "precision_micro": precision_score(labels_masked, preds_masked, average="micro", zero_division=0),
        "recall_micro": recall_score(labels_masked, preds_masked, average="micro", zero_division=0),
        "roc_auc_macro": roc_auc_score(labels_masked, probs_masked, average="macro"),
        "roc_auc_micro": roc_auc_score(labels_masked, probs_masked, average="micro"),
        "ap-score": average_precision_score(labels_masked, probs_masked, average="macro")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest='model_name', type=str, help='Name of used BERT model', default="chandar-lab/NeoBERT")
    parser.add_argument('--model_location', dest='model_location', type=str, help='Location of saved weights', default="neobert")
    args = parser.parse_args()

    dataset = get_test_dataset()

    model_name = args.model_name
    tokenizer = get_tokenizer(model_name)

    model_location = args.model_location
    model = get_model(model_location)

    # Apply the tokenizer to the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
        report_to="none"
    )

    trainer = Trainer(
        model=model,                        # Pre-trained BERT model
        args=training_args,                 # Training arguments
        tokenizer=tokenizer,
        compute_metrics=compute_metrics     # Custom metric
    )

    # Evaluate the model
    results = trainer.evaluate(tokenized_datasets)
    print("RESUlTS ON TEST")
    print(results)

# Initialize the BERT tokenizer
# model_name = "chandar-lab/NeoBERT"
# model_name = "google-bert/bert-base-uncased"
# model_name = "FacebookAI/roberta-base"
# model_name = "nomic-ai/nomic-embed-text-v1-unsupervised"
# model_name = "answerdotai/ModernBERT-base"
main()
