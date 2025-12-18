import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

def remove_identical_ids(results):
    popped = []
    for qid, rels in results.items():
        for pid in list(rels):
            if qid == pid:
                results[qid].pop(pid)
                popped.append(pid)
    return results


def mrr_func(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        return_scores: bool = False,
) -> tuple[dict[str, float]]:
    MRR = {}

    per_query_scores = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    print("\n")

    ## Get top-k_max documents for each query
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
    # Calculate the score for each query
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])

        per_query_scores[query_id] = {}  # Initialize

        for k in k_values:
            rr = 0.0
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    rr = 1.0 / (rank + 1)
                    break
            MRR[f"MRR@{k}"] += rr  # Save the sum of reciprocal rank scores for all queries
            per_query_scores[query_id][f"MRR@{k}"] = rr

    # Normalize and average
    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / len(qrels), 5)
        #print("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    if return_scores:
        return MRR, per_query_scores
    else:
        return MRR

def merge_beir_eval_scores(*score_dicts):
    """
    Merge multiple per-query score dicts into a single mapping.

    Args:
        *score_dicts: Each dict follows {qid: {metric: value}}.

    Returns:
        merged_scores: {qid: {metric1: value1, metric2: value2, ...}}
    """
    merged_scores = {}

    for scores in score_dicts:
        for qid, metrics in scores.items():
            if qid not in merged_scores:
                merged_scores[qid] = {}
            merged_scores[qid].update(metrics)
    return merged_scores

def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100,1000], return_scores=True):
    import pytrec_eval


    '''
    Copied from ResonIR to compute ndcg/map/recall/precision via pytrec_eval.
    Added MRR@k calculation based on custom_metrics_yk.py.
    Returns:
    1. output: averaged ndcg/map/recall/precision/mrr metrics.
    2. final_scores: per-query metric scores for later analysis.
    For example:
    '''

    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0.0} # Only overall MRR supported here (no MRR@k)

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    # oracle reranker evaluation
    sorted_ids = {}
    top_100_ids = {}
    for query_id in results.keys():
        sorted_ids[query_id] = sorted(results[query_id].keys(), key=lambda x: results[query_id][x], reverse=True)
        top_100_ids[query_id] = set(sorted_ids[query_id][:100])
    oracle_results = {}
    for query_id in results.keys():
        oracle_results[query_id] = {}
        for doc_id in results[query_id].keys():
            if doc_id in top_100_ids[query_id] and query_id in qrels and doc_id in qrels[query_id]: # doc is both top 100 and relevant
                oracle_results[query_id][doc_id] = qrels[query_id][doc_id] # pull relevance label from ground truth
            else:
                oracle_results[query_id][doc_id] = 0
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    oracle_scores = evaluator.evaluate(oracle_results) # keep separate so oracle metrics don't overwrite main ones
    oracle_ndcg = {}
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = 0.0
    for query_id in oracle_scores.keys():
        for k in k_values:
            oracle_ndcg[f"Oracle NDCG@{k}"] += oracle_scores[query_id]["ndcg_cut_" + str(k)]
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = round(oracle_ndcg[f"Oracle NDCG@{k}"] / len(oracle_scores), 5)


    if return_scores:
        # custom addition for MRR@k and per-query scores
        MRR_cut_dict, mrr_per_query_scores = mrr_func(qrels, results, k_values, return_scores=return_scores)
        per_query_scores = merge_beir_eval_scores(scores, mrr_per_query_scores) # per-query metrics
        averaged_scores = {**ndcg, **_map, **recall, **precision, **mrr, **oracle_ndcg, **MRR_cut_dict}
        #print(output)
        return averaged_scores, per_query_scores
    else:
        MRR_cut_dict = mrr_func(qrels, results, k_values, return_scores=return_scores)
        averaged_scores = {**ndcg, **_map, **recall, **precision, **mrr, **oracle_ndcg, **MRR_cut_dict}
        #print(output)
        return averaged_scores

'''
How to use this function?

results = retriever.encode_and_retrieve 
if config.dataset == 'arguana':
    results = remove_identical_ids(results)

output_all_score, merged_query_level_scores = calculate_retrieval_metrics(
    results=results, qrels=qrels, return_scores=True
)
'''

# Custom encoder with CLS pooling (same as original code)
class CLSBiEncoder:
    def __init__(self, model_path, trust_remote_code=False, batch_size=64):
        print(f"Initializing encoder for {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code, local_files_only=True)
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
    

class BM25Model:
        def __init__(self, bm25, corpus_ids):
            self.bm25 = bm25
            self.corpus_ids = corpus_ids
        
        def search(self, corpus, queries, top_k, *args, **kwargs):
            results = {}
            for query_id, query_text in queries.items():
                tokenized_query = query_text.lower().split()
                scores = self.bm25.get_scores(tokenized_query)
                top_indices = np.argsort(scores)[::-1][:top_k]
                results[query_id] = {
                    self.corpus_ids[idx]: float(scores[idx]) 
                    for idx in top_indices
                }
            return results
        
        def encode(self, *args, **kwargs):
            # BM25 doesn't use encoding, BEIR needs this method
            pass

        def search_from_files(self, *args, **kwargs):
            # Not implemented for BM25
            pass
