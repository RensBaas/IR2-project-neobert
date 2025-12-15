# Reproducibility Study of NeoBERT

This repository contains code and experiments for our reproducibility study of NeoBERT.

## Table of Contents
- [Overview](#overview)
- [Environment Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Unified Retrieval Evaluation](#unified-retrieval-evaluation)
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
**Training configuration:**
- Model: `bert-base-uncased`
- Dataset: MS MARCO passage
- Batch size: 64
- Passages per query: 16
- Epochs: 3
- Learning rate: 1e-5
- Expected time: ~2-3 hours on H100

You can run training for BERT using the following commands:
```
python -m tevatron.driver.train \
  --output_dir $HOME/model_msmarco \
  --model_name_or_path bert-base-uncased \
  --save_steps 10000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 64 \
  --train_n_passages 8 \
  --dataloader_num_workers 8 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 128 \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --overwrite_output_dir
```
**OR**

```bash
sbatch train_bert.job
```


### NeoBERT Training
**Training configuration:**
- Model: `chandar-lab/NeoBERT`
- Dataset: MS MARCO passage
- Batch size: 32
- Passages per query: 16
- Epochs: 3
- Learning rate: 1e-5
- Expected time: ~5-6 hours on H100

**Note:** NeoBERT requires `trust_remote_code=True` and uses a custom training script to handle remote code automatically (train_neobert_custom.py).

```bash
sbatch train_neobert.job
```

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
- ```--datasets```. String. Dataset names from the names above for BEIR and BRIGHT, again separated by commas. You can mix and match any of the datasets across the benchmarks, the script takes care of the rest.
- ```--statistics```. Bool (optional; default is False). When set to True, statistical testing (paired t-test and effect size) is run at the end of the evaluation. Statistical testing requires evaluation with at least two models and will be disabled if only one model is given (note: the models need to run correctly, so as before if they're not in our list this step could fail).
- ```--seed```. Int (optional; default is 0). A random seed for experiment run. This mostly affects the procedure for typographic error generation in the robustness to noise exploration.
- ```--noise```. Float (optional; default is 0; between 0 and 1). The percentage of words in a query on which a typographic errors is applied.

This is a WIP! The following features have not yet been implemented:
- ~~Compatibility with query-level statistical testing.~~
- ~~In-domain evaluation with MS MARCO.~~

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

## Classsification

Each classification task has its own training and evaluation code. To use these you will need the same dependencies as used for the retrieval training above, except tevatron. Since hyperparameter were kept the same for each model, there are no options to change these when calling the scripts. For additional hyperparameter tuning you need to change them in the python files or add them in there as additional parsable arguments.

### SST-5
To train a BERT model on SST-5 we have two ways: one is using the python notebook named BERT_classification_sst.ipynb, the other is using our normal python script. This can be called directly with:
```
python bert_classification_sst.py \
  --model_name google-bert/bert-base-uncased \
  --save_name bert-base
```

or by using one of our jobs:
```
sbatch sst_jobs/train_roberta_large_sst_class.job
```
Running the jobs also returns the testset performance and models are saved at: `trained_models/<model_name>_sst_3E`.

### Toxic Comment Classification
#### Data
To train the toxic comment classification models you first need to download the dataset from kaggle by hand an place the .csv files in the same folder as the python script.

#### Training
After downloading the data, the BERT models can be trained using the following code:
```
python bert_classification_toxic.py \
  --model_name google-bert/bert-base-uncased \
  --save_name bert-base \
  --dataset_name train.csv
```
or with one of our job files from `toxic_comment_jobs/`:
```
sbatch toxic_comment_jobs/train_bert_toxic_class.job
```
The trained model gets saved at `trained_models/<save_name>_toxic_class_3E`.

#### Evaluation
To evaluate on the toxic comment classification test set, you can use:
```
python bert_classification_toxic_eval.py \
  --model_name google-bert/bert-base-uncased \
  --model_location ./trained_models/bert-base_toxic_class_3E
```
Or one of our job files in `toxic_comment_eval_jobs/`:
```
sbatch toxic_comment_eval_jobs/train_bert_toxic_class.job
```

### Adversarial NLI
Both training and testset evaluation on the testset for adversarial NLI happen in one script. It can be called using:
```
python bert_classification_adverserial.py \
  --model_name google-bert/bert-base-uncased \
  --save_name bert-base
```
Or one of our job files in adv_jobs:
```
sbatch adv_jobs/train_bert_adv_class.job
```
Models are saved at `trained_models_adv/<save_name>_anli_4E`.




## Repository Structure

```
.
├── README.md
├── eval            # all python files and job files needed for evaluation 
├── train           # all python files and job files needed for training 
└── classification  # all python files and job files needed for classification tasks training and evaluation
```
