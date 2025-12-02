from src.metrics.duo import Duo
from src.utils import load_wiki_balance
from beir.datasets.data_loader import GenericDataLoader
import os, re, json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from huggingface_hub import HfApi, ModelFilter
import os

# import voyageai
# class DuoVO(Duo):
#     def __init__(self, step_size=1):
#         super(DuoVO, self).__init__(step_size=step_size)
#         voyageai.api_key_path = 'voyage_ai.txt'
#         self.embedding_model = voyageai.Client()
        
#     def embed_texts(self, texts):
#         return self.embedding_model.embed(
#                     texts, model="voyage-02", input_type="document"
#                 ).embeddings

api = HfApi()
models = api.list_models(
    filter=ModelFilter(author="sentence-transformers")
)
models = list(models)
sbert_models = [m.id.split('/')[-1] for m in models if m.author=='sentence-transformers']

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def check(arr):
    arr = arr.flatten()
    half = int(len(arr)/2)
    if (arr.flatten()[:half]>0).sum() in {0, half}:
        return True
    return False

def acc(arr):
    return max( (arr>0).sum(), (arr<0).sum() )/len(arr)

results = {}
if not os.path.exists('results'):
    os.makedirs('results')
elif os.path.exists('results/metric_validation.json'):
    with open('results/metric_validation.json', 'r') as infile:
        results = json.load(infile)

corpus, queries, qrels = load_wiki_balance(subset='synthetic')
for model_name in sbert_models:
    
    if model_name in results:
        print('already evaluated', model_name)
        continue

    d = Duo(step_size=2, embedding_model=model_name)
    c = defaultdict(lambda: defaultdict(list))
    
    for key in corpus:
        query_id = '_'.join(key.split('_')[0:2])
        perspective = int(key.split('_')[-2].replace('p', ''))
        doc = int(key.split('_')[-1].replace('d', ''))
        c[query_id][perspective].append(corpus[key]['text'])
        c[query_id]['query'] = queries[query_id]
    
    for query_id in tqdm(c.keys()):
        query = c[query_id]['query']
        docs = c[query_id][1] + c[query_id][2]
        embeddings_raw = d.embedding_model.encode(docs)
        d.pca.fit(embeddings_raw)
        embeddings_transformed = d.pca.transform(embeddings_raw).flatten()

        len_1 = len(c[query_id][1])
        len_2 = len(c[query_id][2])
        c[query_id]['accuracy_1'] = acc(embeddings_transformed[:len_1])
        c[query_id]['accuracy_2'] = acc(embeddings_transformed[len_1:])
        c[query_id]['successes'] = (c[query_id]['accuracy_1']*len_1) + (c[query_id]['accuracy_2']*len_2)
        c[query_id]['success'] = check(embeddings_transformed)
        c[query_id]['embeddings'] = embeddings_transformed
        
    suc = np.array([c[query_id]['successes'] for query_id in c]).sum()
    tot = np.array([ len(c[query_id][1])+len(c[query_id][2]) for query_id in c]).sum()
    
    print(model_name, suc/tot)
    results[model_name] = suc/tot
    
    with open('results/metric_validation.json', 'w') as outfile:
        json.dump(results, outfile)