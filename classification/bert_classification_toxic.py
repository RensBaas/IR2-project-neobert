from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback, DebertaV2Tokenizer
import torch.nn as nn
from evaluate import load
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)
import argparse

label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def combine_labels(data):
    data["labels"] = [float(data[col]) for col in label_columns]
    return data

# {'id': '02141412314', 'comment_text': 'Sample comment text', 'toxic': 0, 'severe_toxic': 0, 'obscene': 0, 'threat': 0, 'insult': 0, 'identity_hate': 1, }
def get_dataset(dataset_name):
    dataset = load_dataset("csv", data_files=dataset_name)

    dataset = dataset.map(combine_labels)
    dataset = dataset.remove_columns(label_columns)
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=11)

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


def get_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                trust_remote_code=True,
                                                                num_labels=6, 
                                                                problem_type="multi_label_classification")
    # if model_name != "chandar-lab/NeoBERT" and model_name != "microsoft/deberta-v3-large": # Apperently neobert & deberta do not support this
    #     model.gradient_checkpointing_enable()

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
        "precision_micro": precision_score(labels_masked, preds_masked, average="micro", zero_division=0),
        "recall_micro": recall_score(labels_masked, preds_masked, average="micro", zero_division=0),
        "roc_auc_macro": roc_auc_score(labels_masked, probs_masked, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, help='Name of used dataset (train or test)', default="train.csv")
    parser.add_argument('--model_name', dest='model_name', type=str, help='Name of used BERT model', default="chandar-lab/NeoBERT")
    parser.add_argument('--save_name', dest='save_name', type=str, help='Name of saved weights after training', default="neobert")
    args = parser.parse_args()


    dataset_name = args.dataset_name
    dataset = get_dataset(dataset_name)

    model_name = args.model_name
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)

    # Apply the tokenizer to the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=100,
        fp16=True,
        report_to="none"
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,                        # Pre-trained BERT model
        args=training_args,                 # Training arguments
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],  # Is actually validate
        tokenizer=tokenizer,
        data_collator=data_collator,        # Efficient batching
        compute_metrics=compute_metrics     # Custom metric
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))


    # Start training
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    print(results)

    save_name = args.save_name
    trainer.save_model("./trained_models/{}_toxic_class_3E".format(save_name))


# Initialize the BERT tokenizer
# model_name = "chandar-lab/NeoBERT"
# model_name = "google-bert/bert-base-uncased"
# model_name = "FacebookAI/roberta-base"
# model_name = "FacebookAI/roberta-base"
# model_name = "answerdotai/ModernBERT-base"
main()
