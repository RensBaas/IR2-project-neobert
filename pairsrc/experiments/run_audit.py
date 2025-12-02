from src.utils import load_wiki_balance
from src.metrics.measures import *
from src.metrics.duo import indexical_bias_results
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.sparse import SparseSearch
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
from beir import util
from tqdm import tqdm
from ast import literal_eval
import pandas as pd
import numpy as np
import pathlib, os, json
import argparse
        
def belongs_to_category(_id, category, categories_csv):
    x = categories_csv[categories_csv['id']==int(_id)]
    if len(x):
        return category in x.iloc[0]['categories']
    else:
        return False
 

def write_results(duo_results, bias_results_raw, args, prefix, postfix):
    raw_fn = os.path.join(args.results_dir, f"results_bias_raw_{prefix}_{args.random_state}{postfix}.json")
    clean_fn = os.path.join(args.results_dir, f"results_bias_{args.random_state}{postfix}.json")
    with open(raw_fn, "w") as outfile:
        bias_results_raw['duo_bias'] = list(duo_results)
        bias_results_raw['embedding_model'] = args.embedding_model
        bias_results_raw['step_size'] = args.step_size
        bias_results_raw['random_state'] = args.random_state
        json.dump(bias_results_raw, outfile)
        
    bias_results = {}
    for metric in METRIC_NAMES:
        bias_results[metric] = {}
        for p in ['1', '2']:
            bias_results[metric][p] = f"{np.nanmean(np.array(bias_results_raw[metric][int(p)])):.2f} +-{np.nanstd(np.array(bias_results_raw[metric][int(p)])):.2f}"
    bias_results['duo_bias'] = f"{np.nanmean(duo_results):.2f} +-{np.nanstd(duo_results):.2f}"
    with open(clean_fn, "w") as outfile:
        json.dump(bias_results, outfile)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="sbert",
        choices=["bm25",
                 "sbert",
                 "ance",
                 "dpr",
                 "use-qa",
                 "sparta",
                 "splade",
                 "colbert"
                ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic",
                 "natural"
                ],
    )
    parser.add_argument(
        "--fit_dataset",
        type=str,
        default=None
    )
    parser.add_argument(
        "--spurious_dataset",
        type=str,
        default=None
    )
    parser.add_argument("--categories", type=str, default="categories.csv")
    parser.add_argument("--path_to_Z_score", '-z', type=str)
    parser.add_argument("--overwrite_ranking", action="store_true")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--random_state", type=int, default=7)
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("Loading corpus...")
    corpus, queries, qrels = load_wiki_balance(subset=args.dataset)
    
    fit_corpus, fit_qrels = None, None
    if args.fit_dataset:
        fit_corpus, _, fit_qrels = load_wiki_balance(subset=args.fit_dataset)
        
    spurious_corpus = None
    postfix = ""
    if args.spurious_dataset:
        spurious_corpus, _, _ = load_wiki_balance(subset=args.spurious_dataset)
        postfix = "_s"
    
    model = DRES(models.SentenceBERT("msmarco-roberta-base-ance-firstp"))   
    if args.model == "dpr":
        model = DRES(models.DPR(("facebook/dpr-question_encoder-multiset-base", "facebook/dpr-ctx_encoder-multiset-base")), batch_size=1)
        normalize = False
    elif args.model == "ance":
        model = DRES(models.SentenceBERT("msmarco-roberta-base-ance-firstp"))

    elif args.model == "sbert":
        model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
        normalize = True
    elif args.model == "use-qa":
        model = DRES(models.UseQA("https://tfhub.dev/google/universal-sentence-encoder-qa/3"))
        normalize = False
    elif args.model == "sparta":
        model = SparseSearch(models.SPARTA("BeIR/sparta-msmarco-distilbert-base-v1"), batch_size=128)
    elif args.model == "splade":
        model = DRES(models.SPLADE("naver/splade_v2_distil"), batch_size=128)
    elif args.model == "bm25":
        hostname = "localhost"
        index_name = args.dataset.replace("/", " ") 
        initialize = True
        number_of_shards = 1
        model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

    retriever = EvaluateRetrieval(model)
    if args.model != 'bm25':
        retriever = EvaluateRetrieval(model, score_function="dot")
        
    args.results_dir = f'{args.results_dir}/{args.dataset}/{args.model}'
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    if args.overwrite_ranking or (not os.path.exists(f'{args.results_dir}/retrieved.json')): # retrieve again
        print("Retrieving...")
        retrieved = retriever.retrieve(corpus, queries)
        
        with open(f'{args.results_dir}/retrieved.json', 'w') as outfile:
            json.dump(retrieved, outfile)
        
    with open(f'{args.results_dir}/retrieved.json', 'r') as infile:
        retrieved = json.load(infile)
        
    print("Computing results...")
    ndcg, results, recall, precision = retriever.evaluate(qrels, retrieved, retriever.k_values)
    results.update(ndcg)
    results.update(recall)
    results.update(precision)
    with open( os.path.join(args.results_dir, "results_relevance.json"), "w") as outfile:
        json.dump(results, outfile)
      
    # clean up memory first
    del model
    del retriever
        
    categories_csv = pd.read_csv(args.categories)
    categories_csv["categories"] = [literal_eval(x) for x in categories_csv["categories"].values]
    cats = set()
    for c in categories_csv["categories"].values:
        cats.update(set(c))

    # compute and write indexical bias results for each category
    for cat in cats:
        if cat in {"Religion", "Philosophy"}:
                continue
        retrieved_cat = {query_idx:retrieved[query_idx] for query_idx in retrieved if belongs_to_category(query_idx.split('_')[0], category=cat, categories_csv=categories_csv)}
        if len(retrieved_cat)>1:
            duo_results_cat = indexical_bias_results(retrieved_cat, corpus, qrels, fit_corpus=fit_corpus, fit_qrels=fit_qrels, spurious_corpus=spurious_corpus, path_to_Z_score=args.path_to_Z_score, step_size=args.step_size, embedding_model=args.embedding_model, random_state=args.random_state)
            bias_results_raw_cat = stoyanovich_bias_results(retrieved_cat, qrels)
            write_results(duo_results_cat, bias_results_raw_cat, args, prefix=cat, postfix=postfix)

    # compute and write overall indexical bias results
    duo_results = indexical_bias_results(retrieved, corpus, qrels, fit_corpus=fit_corpus, fit_qrels=fit_qrels, spurious_corpus=spurious_corpus, path_to_Z_score=args.path_to_Z_score, step_size=args.step_size, embedding_model=args.embedding_model, random_state=args.random_state)
    bias_results_raw = stoyanovich_bias_results(retrieved, qrels)
    write_results(duo_results, bias_results_raw, args, prefix="", postfix=postfix)
        
main()