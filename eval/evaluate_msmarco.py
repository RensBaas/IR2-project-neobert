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


# CLS pooling instead of mean pooling that is used in beir
class CLSBiEncoder:
    def __init__(self, model_path, trust_remote_code=False, batch_size=64):
        print(f"Initializing encoder for {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )  # trust_remote_code for NeoBERT
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model.eval()
        self.batch_size = batch_size

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)

    def encode_queries(self, queries, batch_size=None, **kwargs):
        print(f"encoding {len(queries)} queries", flush=True)
        if batch_size is None:
            batch_size = self.batch_size
        return self._encode(queries, batch_size)

    def encode_corpus(self, corpus, batch_size=None, **kwargs):
        print(f"encoding corpus type: {type(corpus)}", flush=True)
        if batch_size is None:
            batch_size = self.batch_size

        sentences = []
        if isinstance(corpus, dict):  # dict
            for doc_id in corpus:
                doc = corpus[doc_id]
                title = doc.get("title", "")
                text = doc.get("text", "")
                sentences.append(title + " " + text if title else text)
        else:  # list
            for doc in corpus:
                title = doc.get("title", "")
                text = doc.get("text", "")
                sentences.append(title + " " + text if title else text)

        print(f"Prepared {len(sentences)} sentences to encode", flush=True)
        return self._encode(sentences, batch_size)

    def _encode(self, sentences, batch_size):
        print(f"Starting encoding of {len(sentences)} sentences with batch size {batch_size}", flush=True)
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
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS

            all_embeddings.append(embeddings)

            if (start_idx // batch_size + 1) % 100 == 0:
                print(f"Encoded {start_idx + len(batch_sentences)}/{len(sentences)} sentences", flush=True)

        print("Encoding complete", flush=True)
        return np.vstack(all_embeddings)


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
bert_results_file = os.path.join(results_dir, f"{dataset}_bert_cls.json")
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
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

    util.save_runfile(os.path.join(results_dir, f"{dataset}_bert_cls.run.trec"), results)
    util.save_results(bert_results_file, ndcg, _map, recall, precision, mrr)

    print("\nBERT Results on MS MARCO:")
    print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
    print(f"MAP@10: {_map['MAP@10']:.4f}")
    print(f"Recall@10: {recall['Recall@10']:.4f}")
    print(f"MRR@10: {mrr['MRR@10']:.4f}", flush=True)

#### Evaluate NeoBERT
print("\n" + "=" * 50)
print("Evaluating NeoBERT on MS MARCO")
print("=" * 50, flush=True)

print("Creating NeoBERT model", flush=True)
model = DRES(CLSBiEncoder("/home/scur1736/model_msmarco_neobert", trust_remote_code=True, batch_size=64))
print("Creating retriever...", flush=True)
retriever = EvaluateRetrieval(model, score_function="dot")

print("Starting retrieval + saving embeddings", flush=True)
results = retriever.encode_and_retrieve(corpus, queries, encode_output_path="/home/scur1736/embeddings/neobert/")

print("Evaluating results...", flush=True)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

util.save_runfile(os.path.join(results_dir, f"{dataset}_neobert_cls.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{dataset}_neobert_cls.json"), ndcg, _map, recall, precision, mrr)

print("\nNeoBERT Results on MS MARCO:")
print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
print(f"MAP@10: {_map['MAP@10']:.4f}")
print(f"Recall@10: {recall['Recall@10']:.4f}")
print(f"MRR@10: {mrr['MRR@10']:.4f}", flush=True)
