from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback, DebertaV2Tokenizer
import torch.nn as nn
from evaluate import load
import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score
)
import argparse


def get_dataset(dataset_name):
    ds = load_dataset(dataset_name)

    # concatenate datasets because we don't care about rounds for this experiment
    train = concatenate_datasets([ds["train_r1"], ds["train_r2"], ds["train_r3"]])
    dev   = concatenate_datasets([ds["dev_r1"],   ds["dev_r2"],   ds["dev_r3"]])
    test  = concatenate_datasets([ds["test_r1"],  ds["test_r2"],  ds["test_r3"]])
    
    dataset = DatasetDict({
        "train": train,
        "validation": dev,
        "test": test
    })

    dataset = dataset.rename_column("label", "labels")

    return dataset


def get_tokenizer(tokenizer_name):
    if tokenizer_name == "microsoft/deberta-v3-large":      
        tokenizer = DebertaV2Tokenizer.from_pretrained(
                "microsoft/deberta-v3-large",
                use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return tokenizer

def tokenize_function(batch, tokenizer):
    return tokenizer(batch["premise"], batch["hypothesis"], padding="max_length", truncation=True, max_length=512)


def get_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=3)
    return model


# Define a custom compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "accuracy": accuracy_score(labels, predictions),
        "precision": recall_score(labels, predictions, average="weighted"),
        "recall": precision_score(labels, predictions, average="weighted")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest='model_name', type=str, help='Name of used BERT model', default="chandar-lab/NeoBERT")
    parser.add_argument('--save_name', dest='save_name', type=str, help='Name of saved weights after training', default="neobert")
    args = parser.parse_args()

    dataset_name = "facebook/anli"
    dataset = get_dataset(dataset_name)

    model_name = args.model_name
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)

    # Apply the tokenizer to the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

    training_args = TrainingArguments(
        output_dir="./results_adv",
        eval_strategy="epoch",
        save_strategy="epoch",
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs_adv",
        logging_steps=100,
        fp16=True,
        report_to="none"
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,                        # Pre-trained BERT model
        args=training_args,                 # Training arguments
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],  # Is actually validate
        tokenizer=tokenizer,
        data_collator=data_collator,        # Efficient batching
        compute_metrics=compute_metrics     # Custom metric
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))


    # Start training
    trainer.train()


    # Evaluate the model on val
    results = trainer.evaluate(tokenized_datasets["validation"])
    print("RESULTS ON VALIDATION")
    print(results)

    # Evaluate the model on test
    results = trainer.evaluate(tokenized_datasets["test"])
    print("RESULTS ON TEST")
    print(results)

    save_name = args.save_name
    trainer.save_model("./trained_models_adv/{}_anli_4E".format(save_name))


# Initialize the BERT tokenizer
# model_name = "chandar-lab/NeoBERT"
# model_name = "google-bert/bert-base-uncased"
# model_name = "FacebookAI/roberta-base"
# model_name = "answerdotai/ModernBERT-base"
main()
