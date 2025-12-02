import json
import logging
import os

import numpy as np
import torch
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
    force=True,
)


# Custom encoder with CLS pooling (same as original code)
class CLSBiEncoder:
    def __init__(self, model_path, trust_remote_code=False, batch_size=64):
        print(f"Initializing encoder for {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model.eval()
        self.batch_size = batch_size

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)

    def encode_queries(self, queries, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        return self._encode(queries, batch_size)

    def encode_corpus(self, corpus, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size

        sentences = []
        if isinstance(corpus, dict):
            for doc_id in corpus:
                doc = corpus[doc_id]
                title = doc.get("title", "")
                text = doc.get("text", "")
                sentences.append(title + " " + text if title else text)
        else:
            for doc in corpus:
                title = doc.get("title", "")
                text = doc.get("text", "")
                sentences.append(title + " " + text if title else text)

        return self._encode(sentences, batch_size)

    def _encode(self, sentences, batch_size):
        all_embeddings = []

        for start_idx in range(0, len(sentences), batch_size):
            batch_sentences = sentences[start_idx : start_idx + batch_size]

            encoded = self.tokenizer(
                batch_sentences, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )

            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


def evaluate_model_on_dataset(model_name, model_path, dataset_name, trust_remote_code=False):
    """Evaluate a model on BEIR dataset"""

    print(f"\n{'=' * 70}")
    print(f"Evaluating {model_name} on {dataset_name}")
    print(f"{'=' * 70}\n")

    # Download dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, "/home/scur1736/datasets")

    # Load data
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"Loaded {len(corpus)} docs, {len(queries)} queries")

    embeddings_dir = f"/home/scur1736/embeddings/{model_name.lower()}/{dataset_name}/"
    os.makedirs(embeddings_dir, exist_ok=True)

    # Create model
    model = DRES(CLSBiEncoder(model_path, trust_remote_code=trust_remote_code, batch_size=64))
    retriever = EvaluateRetrieval(model, score_function="dot")

    # Retrieve
    print("Encoding and retrieving")
    results = retriever.encode_and_retrieve(corpus, queries, encode_output_path=embeddings_dir)

    # Evaluate
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

    # Print results
    print(f"\n{model_name} Results on {dataset_name}:")
    print(f"  NDCG@10:  {ndcg['NDCG@10']:.4f}")
    print(f"  MAP@10:   {_map['MAP@10']:.4f}")
    print(f"  Recall@10: {recall['Recall@10']:.4f}")
    print(f"  MRR@10:   {mrr['MRR@10']:.4f}")

    return {
        "ndcg@10": ndcg["NDCG@10"],
        "map@10": _map["MAP@10"],
        "recall@10": recall["Recall@10"],
        "mrr@10": mrr["MRR@10"],
    }


# Datasets to evaluate on
datasets = [
    "scifact",  #  scientific IR
    "nfcorpus",  # medical IR
    "arguana",  # argument retrieval
    "fiqa",  # financial IR
    "trec-covid",  # COVID (medical) IR
]

# Models to evaluate
models = [
    ("BERT", "/home/scur1736/model_msmarco", False),
    ("NeoBERT", "/home/scur1736/model_msmarco_neobert", True),
]

# Store all results
all_results = {model_name: {} for model_name, _, _ in models}

# Evaluate each model on each dataset
for dataset in datasets:
    for model_name, model_path, trust_remote in models:
        try:
            results = evaluate_model_on_dataset(model_name, model_path, dataset, trust_remote)
            all_results[model_name][dataset] = results
        except Exception as e:
            print(f"Error evaluating {model_name} on {dataset}: {str(e)}")
            all_results[model_name][dataset] = None

# Save results
results_dir = "/home/scur1736/results"
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, "beir_comparison.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# summary table
print("\n" + "=" * 80)
print("SUMMARY: BERT vs NeoBERT on BEIR Datasets")
print("=" * 80)
print(f"\n{'Dataset':<15} {'BERT NDCG@10':<15} {'NeoBERT NDCG@10':<15} {'Improvement':<15}")
print("-" * 60)

bert_scores = []
neobert_scores = []

for dataset in datasets:
    if all_results["BERT"].get(dataset) and all_results["NeoBERT"].get(dataset):
        bert_score = all_results["BERT"][dataset]["ndcg@10"]
        neobert_score = all_results["NeoBERT"][dataset]["ndcg@10"]
        improvement = ((neobert_score - bert_score) / bert_score) * 100

        bert_scores.append(bert_score)
        neobert_scores.append(neobert_score)

        print(f"{dataset:<15} {bert_score:<15.4f} {neobert_score:<15.4f} {improvement:+.1f}%")

if bert_scores and neobert_scores:
    avg_bert = np.mean(bert_scores)
    avg_neobert = np.mean(neobert_scores)
    avg_improvement = ((avg_neobert - avg_bert) / avg_bert) * 100

    print("-" * 60)
    print(f"{'AVERAGE':<15} {avg_bert:<15.4f} {avg_neobert:<15.4f} {avg_improvement:+.1f}%")

print("\n" + "=" * 80)
print(f"Results saved to: {results_dir}/beir_comparison.json")
print("=" * 80)
