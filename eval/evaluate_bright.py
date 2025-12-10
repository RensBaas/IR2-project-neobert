import json
import logging
import os
import traceback

import numpy as np
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from eval_utils import calculate_retrieval_metrics
from utils import CLSBiEncoder

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
    force=True,
)


def evaluate_model_on_dataset(model_name, model_path, dataset_name, trust_remote_code=False):
    """Evaluate a model on BRIGHT dataset"""

    print(f"\n{'=' * 70}")
    print(f"Evaluating {model_name} on {dataset_name}")
    print(f"{'=' * 70}\n")

    # Download dataset
    url = f"https://github.com/liyongkang123/extended_beir_datasets/releases/download/beir_v1.0/{dataset}.zip"
    data_path = util.download_and_unzip(url, "/home/scur1736/datasets")

    # Load data
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"Loaded {len(corpus)} docs, {len(queries)} queries")

    # Create model
    model = DRES(CLSBiEncoder(model_path, trust_remote_code=trust_remote_code, batch_size=64))
    retriever = EvaluateRetrieval(model, score_function="dot")

    # Retrieve
    print("Encoding and retrieving")
    results = retriever.retrieve(corpus, queries)

    print("Calculating metrics")
    output_all_scores, query_level_scores = calculate_retrieval_metrics(
        results=results, qrels=qrels, return_scores=True
    )

    results_dir = f"/home/scur1736/results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, f"{model_name.lower()}_metrics.json"), "w") as f:
        json.dump(output_all_scores, f, indent=2)

    with open(os.path.join(results_dir, f"{model_name.lower()}_query_scores.json"), "w") as f:
        json.dump(query_level_scores, f, indent=2)

    print(f"\n{model_name} Results on {dataset_name}:")
    print(f"  NDCG@10:        {output_all_scores['NDCG@10']:.4f}")
    print(f"  MAP@10:         {output_all_scores['MAP@10']:.4f}")
    print(f"  Recall@10:      {output_all_scores['Recall@10']:.4f}")
    print(f"  MRR@10:         {output_all_scores['MRR@10']:.4f}")
    print(f"  Oracle NDCG@10: {output_all_scores['Oracle NDCG@10']:.4f}")

    return {
        "ndcg@10": output_all_scores["NDCG@10"],
        "map@10": output_all_scores["MAP@10"],
        "recall@10": output_all_scores["Recall@10"],
        "mrr@10": output_all_scores["MRR@10"],
        "oracle_ndcg@10": output_all_scores["Oracle NDCG@10"],
    }


# BRIGHT datasets
datasets = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
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
            print(f"\nError evaluating {model_name} on {dataset}: {str(e)}")
            traceback.print_exc()
            all_results[model_name][dataset] = None

# Save results
results_dir = "/home/scur1736/results"
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, "bright_comparison.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# Print summary table
print("\n" + "=" * 90)
print("SUMMARY: BERT vs NeoBERT on All BEIR Datasets")
print("=" * 90)
print(f"\n{'Dataset':<20} {'BERT NDCG@10':<15} {'NeoBERT NDCG@10':<15} {'Improvement':<15}")
print("-" * 70)

bert_scores = []
neobert_scores = []

for dataset in datasets:
    if all_results["BERT"].get(dataset) and all_results["NeoBERT"].get(dataset):
        bert_score = all_results["BERT"][dataset]["ndcg@10"]
        neobert_score = all_results["NeoBERT"][dataset]["ndcg@10"]
        improvement = ((neobert_score - bert_score) / bert_score) * 100

        bert_scores.append(bert_score)
        neobert_scores.append(neobert_score)

        print(f"{dataset:<20} {bert_score:<15.4f} {neobert_score:<15.4f} {improvement:+.1f}%")

if bert_scores and neobert_scores:
    avg_bert = np.mean(bert_scores)
    avg_neobert = np.mean(neobert_scores)
    avg_improvement = ((avg_neobert - avg_bert) / avg_bert) * 100

    print("-" * 70)
    print(f"{'AVERAGE':<20} {avg_bert:<15.4f} {avg_neobert:<15.4f} {avg_improvement:+.1f}%")

print("\n" + "=" * 90)
print(f"Results saved to: {results_dir}/bright_comparison.json")
print(f"Per-dataset results in: {results_dir}/<dataset>/")
print("=" * 90)
