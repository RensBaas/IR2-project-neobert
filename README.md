# Reproducibility Study of NeoBERT

This repository contains code and experiments for our reproducibility study of NeoBERT.

## Table of Contents
- [Overview](#overview)
- [Environment Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- Unified Retrieval Evaluation
- [Results](#results)
- [Repository Structure](#repository-structure)

## Overview

We fine-tune both BERT-base-uncased and NeoBERT models on the MS MARCO passage ranking dataset and evaluate their performance on:
- MS MARCO dev set (MRR@10)
- 13 BEIR benchmark datasets (NDCG@10)
- 12 BRIGHT benchmark datasets (NDCG@10)

Statistical significance is assessed using paired t-tests at the query level.

## Environment Installation

### 1. Create Virtual Environment

```bash
python -m venv $HOME/tevatron_env
source $HOME/tevatron_env/bin/activate
```

### 2. Install Training Dependencies

```bash
pip install --upgrade pip
pip install tevatron
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install xformers

# Fix dependency conflicts
pip install transformers==4.35.0 tokenizers==0.19.1 huggingface-hub==0.25.0 --force-reinstall
```

### 3. Install Evaluation Dependencies

**Important note**: install these dependencies **AFTER** training models!!!

```bash
pip install beir
pip install faiss-cpu  # faiss gpu out of memory 
pip install "numpy<2"  # Required for faiss compatibility
pip install fire scipy rank-bm25 shortuuid textattack tqdm # Requirements for unified evaluation
```

## Training - train folder 

### BERT Training

```bash
sbatch train_bert.job
```

**Training configuration:**
- Model: `bert-base-uncased`
- Dataset: MS MARCO passage
- Batch size: 64
- Passages per query: 16
- Epochs: 3
- Learning rate: 1e-5
- Expected time: ~2-3 hours on H100

### NeoBERT Training

```bash
sbatch train_neobert.job
```

**Training configuration:**
- Model: `chandar-lab/NeoBERT`
- Dataset: MS MARCO passage
- Batch size: 32
- Passages per query: 16
- Epochs: 3
- Learning rate: 1e-5
- Expected time: ~5-6 hours on H100

**Note:** NeoBERT requires `trust_remote_code=True` and uses a custom training script to handle remote code automatically (train_neobert_custom.py).

## Evaluation - eval file

### MS MARCO Evaluation

```bash
sbatch eval_msmarco.job
```

This will:
1. Download MS MARCO dev set
2. Encode and save embeddings for corpus and queries for both models
3. Perform retrieval and calculate metrics
4. Save results

**Expected time:** ~2-3 hours per model

### BEIR Evaluation

```bash
sbatch eval_beir.job
```

Evaluates both models on BEIR datasets:
- trec-covid
- nfcorpus
- nq
- hotpotqa
- fiqa
- arguana
- webis-touche2020
- quora
- dbpedia-entity
- scidocs
- fever
- climate-fever
- scifact

### BRIGHT Evaluation

```bash
sbatch eval_bright.job
```

Evaluates both models on BRIGHT datasets:
- biology
- earth_science
- economics
- psychology
- robotics
- stackoverflow
- sustainable_living
- leetcode
- pony
- aops
- theoremqa_questions
- theoremqa_theorems


### Statistical Significance Testing

```bash
sbatch tests.job
```

Performs paired t-tests comparing BERT and NeoBERT on all datasets:
- Uses MRR@10 for MS MARCO
- Uses NDCG@10 for BEIR and BRIGHT datasets
- Reports p-values and Cohen's d effect sizes

### Unified Retrieval Evaluation
```python eval/evaluate_retrieval.py run --base_dir [path/to/folder] --models [model1,model2,...] --datasets [dataset1,dataset2,...] --seed [random_seed] --noise [percentage]```

We offer a unified retrieval evaluation script for BEIR and BRIGHT. The command has the following options:
- ```--base_dir```. String. The folder where experiments files will be stored. The script builds the folder structure for each experiment run automatically.
- ```--models```. String. Model names should be delineated by commas (and no spaces!). The script is known to support the following retrieval options ```bert, neobert, bm25```, but theoretically should be able to support other BERT-like encoders from HuggingFace if passing the full model name.
- ```--datasets``` String. Dataset names from the names above for BEIR and BRIGHT, again separated by commas. You can mix and match any of the datasets across the benchmarks, the script takes care of the rest.
- ```--seed```. Int (optional; default is 0). A random seed for experiment run. This mostly affects the procedure for typographic error generation in the robustness to noise exploration.
- ```--noise```. Float (optional; default is 0; between 0 and 1). The percentage of words in a query on which a typographic errors is applied.

This is a WIP! The following features have not yet been implemented:
- Compatibility with query-level statistical testing.
- In-domain evaluation with MS MARCO.

## Results

Results are saved in `/home/<user>/results/`:

```
results/
├── msmarco_bert_cls_detailed.json
├── msmarco_bert_cls_query_scores.json
├── msmarco_neobert_cls_detailed.json
├── msmarco_neobert_cls_query_scores.json
├── statistical_tests_results.json
├── beir_comparison.json
├── bright_comparison.json
└── <dataset>/
    ├── bert_metrics.json
    ├── bert_query_scores.json
    ├── neobert_metrics.json
    └── neobert_query_scores.json
```

See paper for detailed analysis.

## Repository Structure

```
.
├── README.md
├── eval         # all python files and job files needed for evaluation 
└── train        # all python files and job files needed for training 
```
