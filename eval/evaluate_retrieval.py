import json
import logging
import os
import fire

from eval_utils import CLSBiEncoder, BM25Model, calculate_retrieval_metrics, remove_identical_ids
from make_typo_queries import make_noisy_queries
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

from rank_bm25 import BM25Okapi
import shortuuid
import re
import shutil
import random
import time

MODEL_CHECKPOINTS = { # Update with HF checkpoints when public
    "bert": ("/home/scur1736/model_msmarco", False),
    "neobert": ("/home/scur1736/model_msmarco_neobert", True),
}

BEIR = ["trec-covid","nfcorpus","nq","hotpotqa","fiqa","arguana","webis-touche2020","quora","dbpedia-entity",
        "scidocs","fever","climate-fever","scifact"]
BRIGHT = ["biology","earth_science","economics","psychology","robotics","stackoverflow","sustainable_living",
          "leetcode","pony","aops","theoremqa_questions","theoremqa_theorems"]

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
    force=True,
)

class RetrievalExperiment:
    def __init__(self, base_dir, models, datasets, seed=0, noise=0):
        self.run = shortuuid.uuid()

        # Build folder structure
        self.exp_dir = Path(base_dir) / f'Run_{self.run}'

        self.data_path = Path(self.exp_dir) / "datasets"
        os.makedirs(self.data_path, exist_ok=True)

        self.embeddings_path = Path(self.exp_dir) / "embeddings"
        os.makedirs(self.embeddings_path, exist_ok=True)

        self.results_path = Path(self.exp_dir) / 'results'
        os.makedirs(self.results_path, exist_ok=True)

        if isinstance(models, str):
            models = models.split(",")
        if isinstance(datasets, str):
            datasets = datasets.split(",")

        # Download and unzip datasets
        self.datasets = datasets
        self.mixed = (len(set(self.datasets).intersection(set(BEIR))) > 0) and (len(set(self.datasets).intersection(set(BRIGHT))) > 0)

        for dataset_name in self.datasets:
            if dataset_name in BEIR:
                url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            elif dataset_name in BRIGHT:
                url = f"https://github.com/liyongkang123/extended_beir_datasets/releases/download/beir_v1.0/{dataset_name}.zip"
            else:
                print(f'{dataset_name} is not a known dataset. Removing {dataset_name} from evaluation.')
                self.datasets.pop(dataset_name)
                continue
            _ = util.download_and_unzip(url, str(self.data_path))

        # Copy datasets and add noise
        if noise > 0:
            self.noisy_data_path = self.exp_dir / "datasets_noisy"
            os.makedirs(self.noisy_data_path, exist_ok=True)
            for f in self.data_path.glob("*.zip"):
                f.unlink()  
            shutil.copytree(self.data_path, self.noisy_data_path, dirs_exist_ok=True)
            
            for dataset_name in self.datasets:
                make_noisy_queries(self.noisy_data_path / dataset_name / "queries.jsonl",
                                   self.noisy_data_path / dataset_name / "queries.jsonl",
                                   noise_level=noise)

        # Get model name, path, and trust_remote
        self.models = dict()
        for m in models:
            name, (path, trust_rem) = self.get_model(m)
            self.models[name] = (path, trust_rem)
        
        # Make embedding folders
        for dataset_name in self.datasets:
            for model_name in self.models.keys():
                os.makedirs(self.embeddings_path / model_name / dataset_name, exist_ok=True)

        self.results_buffer = {model_name: {} for model_name in self.models.keys()}
        self.experiment_metadata = {
            'timestamp': time.asctime(),
            'run_id': self.run,
            'models': ','.join(list(self.models.keys())),
            'datasets': ','.join(self.datasets),
            'seed': str(seed),
            'noise': str(noise),
            'experiment_dir': str(self.exp_dir),
        }

    def run_experiment(self):
        for dataset in self.datasets:
            for model in self.models.items():
                model_name, (model_path, trust_remote) = model
                try:
                    dataset_dir = self.data_path / dataset
                    embbedings_dir = self.embeddings_path / model_name / dataset
                    results, full_average, full_per_query = RetrievalExperiment._evaluate_model_on_dataset(model_name, 
                                                        model_path, 
                                                        dataset_dir, 
                                                        embbedings_dir, 
                                                        trust_remote)
                    self.results_buffer[model_name][dataset] = results
                except Exception as e:
                    print(f"Error evaluating {model_name} on {dataset}: {str(e)}")
                    self.results_buffer[model_name][dataset] = None

                self._save_round_results(model_name, dataset, full_average, full_per_query)

        self._save_results()

    def _save_round_results(self, model_name, dataset_name, average_results, per_query_results):
        round_metadata = self.experiment_metadata.copy()
        round_metadata['datasets'] = dataset_name
        round_metadata['models'] = model_name
        with open(self.results_path / f"{model_name}_{dataset_name}_average_scores.json", "w") as f:
            json.dump({'metadata': round_metadata,
                       'average_results' : average_results},
                       f, 
                       indent=2)
            
        with open(self.results_path / f"{model_name}_{dataset_name}_query_scores", "w") as f:
            json.dump({'metadata': round_metadata,
                       'per_query_results' : per_query_results},
                       f, 
                       indent=2)
        

    def _save_results(self):
        if self.mixed:
            beir_buffer, bright_buffer = dict(), dict()
            for model, datasets in self.results_buffer.items():
                for dataset, results in datasets.items():
                    if dataset in BEIR:
                        beir_buffer.setdefault(model, {})[dataset] = results
                    else:
                        bright_buffer.setdefault(model, {})[dataset] = results
            with open(self.results_path / "retrieval_comparison.json", "w") as f:
                json.dump({'metadata': self.experiment_metadata,
                        'beir_results' : beir_buffer,
                        'bright_results': bright_buffer},
                        f, 
                        indent=2)
        else:
            with open(self.results_path / "retrieval_comparison.json", "w") as f:
                json.dump({'metadata': self.experiment_metadata,
                        'results' : self.results_buffer},
                        f, 
                        indent=2)

        print(f"Results saved to {self.results_path / 'retrieval_comparison.json'}")

    @staticmethod
    def get_model(model):
        if model in MODEL_CHECKPOINTS.keys():
            return model, MODEL_CHECKPOINTS[model]
        elif model == "bm25":
            return "bm25", (None, False)
        else:
            return RetrievalExperiment._clean_model_name(model), (model, True)
    
    @staticmethod
    def _clean_model_name(model_path):
        model_name = model_path.split('/')[-1]
        model_name = re.sub(r'[^a-zA-Z0-9]', '', model_name)
        model_name = model_name.lower()
        return model_name

    @staticmethod
    def _evaluate_model_on_dataset(model_name, model_path, data_path, embeddings_dir, trust_remote_code=False):
        """Evaluate a model on BEIR dataset"""

        print(f"\n{'=' * 70}")
        print(f"Evaluating {model_name} on {data_path.name}")
        print(f"{'=' * 70}\n")

        # Load data
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        print(f"Loaded {len(corpus)} docs, {len(queries)} queries")

        # Create model
        if model_name == "bm25":
            # Tokenize corpus
            corpus_ids = list(corpus.keys())
            tokenized_corpus = [
                (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).lower().split() 
                for doc_id in corpus_ids
            ]

            # Initialize BM25
            bm25 = BM25Okapi(tokenized_corpus)
            model = BM25Model(bm25, corpus_ids)
            retriever = EvaluateRetrieval(model, score_function="bm25")

            # Retrieve
            print("Encoding and retrieving")
            results = model.search(corpus, queries, top_k=1000)
        else:
            model = DRES(CLSBiEncoder(model_path, trust_remote_code=trust_remote_code, batch_size=64))
            retriever = EvaluateRetrieval(model, score_function="dot")

            # Retrieve
            print("Encoding and retrieving")
            results = retriever.encode_and_retrieve(corpus, queries, encode_output_path=embeddings_dir)

        # Evaluate
        if data_path.name == "arguana":
            print("Removing identical query-doc IDs for arguana")
            results = remove_identical_ids(results)

        print("Calculating metrics")
        averaged_scores, query_level_scores = calculate_retrieval_metrics(
            results=results, qrels=qrels, return_scores=True
        )

        metrics10 = {
            "ndcg@10": averaged_scores["NDCG@10"],
            "map@10": averaged_scores["MAP@10"],
            "recall@10": averaged_scores["Recall@10"],
            "mrr@10": averaged_scores["MRR@10"],
            "oracle_ndcg@10": averaged_scores["Oracle NDCG@10"],
        }

        print(f"\n{model_name} Results on {data_path.name}:")
        print(f"  NDCG@10:        {averaged_scores['NDCG@10']:.4f}")
        print(f"  MAP@10:         {averaged_scores['MAP@10']:.4f}")
        print(f"  Recall@10:      {averaged_scores['Recall@10']:.4f}")
        print(f"  MRR@10:         {averaged_scores['MRR@10']:.4f}")
        print(f"  Oracle NDCG@10: {averaged_scores['Oracle NDCG@10']:.4f}")

        return metrics10, averaged_scores, query_level_scores 


def run_experiment(base_dir, models, datasets, seed=0, noise=0):
    random.seed(seed)
    experiment = RetrievalExperiment(base_dir, models, datasets, seed, noise)
    print(f'{"=" * 70}\nStarting experiment {experiment.run}\n{"=" * 70}')
    experiment.run_experiment()

if __name__ == "__main__":
    fire.Fire({
        "run": run_experiment,
    })