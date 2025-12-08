import json
import logging
import os

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from eval_utils import calculate_retrieval_metrics, remove_identical_ids
from utils import CLSBiEncoder

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
    force=True,
)

print("Loading dataset", flush=True)
# MS MARCO dataset
dataset = "msmarco"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = "/home/scur1736/datasets"
data_path = util.download_and_unzip(url, out_dir)

print("Loading corpus, queries, qrels", flush=True)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")  # dev for MSMARCO
print(f"Loaded {len(corpus)} docs, {len(queries)} queries", flush=True)

results_dir = "/home/scur1736/results"
os.makedirs(results_dir, exist_ok=True)

#### Evaluate BERT
print("\n" + "=" * 50)
print("Evaluating BERT on MS MARCO")
print("=" * 50, flush=True)

# Check if BERT results already exist
bert_results_file = os.path.join(results_dir, f"{dataset}_bert_cls_detailed.json")
bert_query_scores_file = os.path.join(results_dir, f"{dataset}_bert_cls_query_scores.json")

if os.path.exists(bert_results_file):
    print("BERT results already exist, skipping BERT evaluation", flush=True)
else:
    print("Creating BERT model", flush=True)
    model = DRES(CLSBiEncoder("/home/scur1736/model_msmarco", batch_size=64))
    print("Creating retriever", flush=True)
    retriever = EvaluateRetrieval(model, score_function="dot")

    print("Starting retrieval + saving embeddings", flush=True)
    results = retriever.encode_and_retrieve(corpus, queries, encode_output_path="/home/scur1736/embeddings/bert/")

    print("Evaluating results", flush=True)
    # Use custom evaluation function
    output_all_scores, query_level_scores = calculate_retrieval_metrics(
        results=results, qrels=qrels, return_scores=True
    )

    # Save overall metrics
    with open(bert_results_file, "w") as f:
        json.dump(output_all_scores, f, indent=2)

    # Save per-query scores
    with open(bert_query_scores_file, "w") as f:
        json.dump(query_level_scores, f, indent=2)

    print("\nBERT Results on MS MARCO:")
    print(f"NDCG@10: {output_all_scores['NDCG@10']:.4f}")
    print(f"MAP@10: {output_all_scores['MAP@10']:.4f}")
    print(f"Recall@10: {output_all_scores['Recall@10']:.4f}")
    print(f"MRR@10: {output_all_scores['MRR@10']:.4f}")
    print(f"Oracle NDCG@10: {output_all_scores['Oracle NDCG@10']:.4f}", flush=True)


#### Evaluate NeoBERT
print("\n" + "=" * 50)
print("Evaluating NeoBERT on MS MARCO")
print("=" * 50, flush=True)

neobert_results_file = os.path.join(results_dir, f"{dataset}_neobert_cls_detailed.json")
neobert_query_scores_file = os.path.join(results_dir, f"{dataset}_neobert_cls_query_scores.json")


print("Creating NeoBERT model", flush=True)
model = DRES(CLSBiEncoder("/home/scur1736/model_msmarco_neobert", trust_remote_code=True, batch_size=64))
print("Creating retriever", flush=True)
retriever = EvaluateRetrieval(model, score_function="dot")

print("Starting retrieval + saving embeddings", flush=True)
results = retriever.encode_and_retrieve(corpus, queries, encode_output_path="/home/scur1736/embeddings/neobert/")

if dataset == "arguana":
    results = remove_identical_ids(results)

print("Evaluating results", flush=True)
# Use custom evaluation function
output_all_scores, query_level_scores = calculate_retrieval_metrics(results=results, qrels=qrels, return_scores=True)

# Save overall metrics
with open(neobert_results_file, "w") as f:
    json.dump(output_all_scores, f, indent=2)

# Save per-query scores
with open(neobert_query_scores_file, "w") as f:
    json.dump(query_level_scores, f, indent=2)

print("\nNeoBERT Results on MS MARCO:")
print(f"NDCG@10: {output_all_scores['NDCG@10']:.4f}")
print(f"MAP@10: {output_all_scores['MAP@10']:.4f}")
print(f"Recall@10: {output_all_scores['Recall@10']:.4f}")
print(f"MRR@10: {output_all_scores['MRR@10']:.4f}")
print(f"Oracle NDCG@10: {output_all_scores['Oracle NDCG@10']:.4f}", flush=True)
