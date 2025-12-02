import os, re, json, argparse
import pandas as pd
import numpy as np
from glob import glob
import statsmodels.api as sm
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from src.metrics.duo import Duo
from src.utils import load_wiki_balance

def any_links(row, platform):
    prefix = ''
    if platform == 'mturk':
        prefix = 'Answer.'
    count = 0
    for i in range(1, 25):
        col = f'{prefix}url_{i}_log'
        if col in row and type(row[col])==str:
            count += len(re.findall(";", row[col]))
    return count

def get_prior_posterior(row, platform):
    if platform == 'prolific':
        return row['prior'], row['posterior']
    prior, posterior = 0, 0
    for i in range(1, 4):
        if row[f'Answer.prior_agree_{i}.on']:
            prior = i
        elif row[f'Answer.prior_disagree_{i}.on']:
            prior = -i
        if row[f'Answer.posterior_agree_{i}.on']:
            posterior = i
        elif row[f'Answer.posterior_disagree_{i}.on']:
            posterior = -i
    return prior, posterior

def embed_dfs(embedding_model='all-MiniLM-L6-v2', step_size=1, random_state=7, use_spurious_correction=False):
    fit_corpus, fit_queries, fit_qrels = load_wiki_balance(subset="synthetic")
    corpus, queries, qrels = load_wiki_balance(subset="natural")
    spurious_corpus, _, _ = load_wiki_balance(subset="natural")
    
    df_1 = pd.concat([pd.read_csv(fn) for fn in glob(f'hit/seme/output_prolific/outputs_synthetic*.csv')])
    df_2 = pd.concat([pd.read_csv(fn) for fn in glob(f'hit/seme/output_prolific/outputs*.csv') if ('synthetic' not in fn)])
    platforms = ['prolific', 'prolific']
    d = Duo(embedding_model=embedding_model,step_size=step_size, random_state=random_state)

    for i, df in enumerate([df_1, df_2]):
        pp = [get_prior_posterior(row, platforms[i]) for _, row in df.iterrows()]
        df['prior'] = [x[0] for x in pp]
        df['posterior'] = [x[1] for x in pp]
        df['which'] = [x.split("_")[-1] for x in df[f'index'].values]
        df['query_id'] = ['_'.join(x.split("_")[:-1]) for x in df[f'index'].values]
        df['clicked_links'] = [int(any_links(row, platforms[i])) for _, row in df.iterrows()]

        nDCVs = []
        signs = []
        for _, row in tqdm(df.iterrows(), total=len(df)):

            embedding_key = "embedding"
            ranking_key = 'doc_id'

            fit_corpus_relevant_subset = {doc_id: fit_corpus[doc_id] for doc_id in fit_corpus if ( (doc_id in fit_qrels[row['query_id']]) and (fit_qrels[row['query_id']][doc_id]>=4) )}        
            if i == 0: # synthetic
                corpus_relevant_subset = fit_corpus_relevant_subset
            else: # natural corpus
                corpus_relevant_subset = {doc_id: corpus[doc_id] for doc_id in corpus if ( (doc_id in qrels[row['query_id']]) and (qrels[row['query_id']][doc_id]>=4) )}

            test_ranking = [x for x in [row[f'{ranking_key}_{i}'] for i in range(1,15)] if (type(x)==str) and x in corpus_relevant_subset]
            
            if use_spurious_correction:
                #spurious_docs_1 = [ doc_id for doc_id in spurious_corpus if row['query_id']+'_p1' in doc_id]
                spurious_docs_1 = [ doc_id for doc_id in spurious_corpus if (row['query_id'] in doc_id) and ('q1' in doc_id)]
                spurious_docs_1 = {doc_id: spurious_corpus[doc_id] for doc_id in spurious_corpus if doc_id in spurious_docs_1}
                #spurious_docs_2 = [ doc_id for doc_id in spurious_corpus if row['query_id']+'_p2' in doc_id]
                spurious_docs_2 = [ doc_id for doc_id in spurious_corpus if (row['query_id'] in doc_id) and ('q2' in doc_id)]
                spurious_docs_2 = {doc_id: spurious_corpus[doc_id] for doc_id in spurious_corpus if doc_id in spurious_docs_2}
                spurious_docs = []
                if len(spurious_docs_1) and len(spurious_docs_2):
                    #print(".", end="")
                    spurious_docs = [spurious_docs_1, spurious_docs_2]
                    d.embed(transform_docs=corpus_relevant_subset, 
                                fit_docs=fit_corpus_relevant_subset,
                                spurious_docs=spurious_docs
                           )
                    nDCVs.append(
                        d.Duo_batch([test_ranking], use_sign=True)[0]
                    )
                    
                    # determine sign orientation
                    d.embed(transform_docs=fit_corpus_relevant_subset, 
                                fit_docs=None)
                    gt = [(d.embeddings[x] > 0) for x in d.embeddings.keys() if 'p1' in x] # is embedding greater than zero
                    sign = 1
                    if sum(gt)/len(gt)<0.5:
                        sign = -1
                    signs.append(sign)
                    
                else:
                    nDCVs.append(None)
                    signs.append(None)
            else:
                d.embed(transform_docs=corpus_relevant_subset, 
                            fit_docs=fit_corpus_relevant_subset,
                       )
                nDCVs.append(
                    d.Duo_batch([test_ranking], use_sign=True)[0]
                )

                # determine sign orientation
                d.embed(transform_docs=fit_corpus_relevant_subset, 
                            fit_docs=None)
                gt = [(d.embeddings[x] > 0) for x in d.embeddings.keys() if 'p1' in x] # is embedding greater than zero
                sign = 1
                if sum(gt)/len(gt)<0.5:
                    sign = -1
                signs.append(sign)

        df['sign'] = signs
        df['nDCV'] = nDCVs
        df['nDCVs'] = df['nDCV']*df['sign']
        df['nclk'] = (df['clicked_links']>0)*(df['nDCVs'])
    return df_1, df_2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", type=str, default="sentence-t5-xl")
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=7)
    parser.add_argument("--use_spurious_correction", action="store_true")
    return parser.parse_args()

def main():
    args = parse_arguments()
    postfix = ""
    if args.use_spurious_correction:
        postfix="_s"
    fn_1 = f"results/seme/synthetic_{args.embedding_model}_{args.step_size}_{args.random_state}{postfix}.csv"
    fn_2 = f"results/seme/natural_{args.embedding_model}_{args.step_size}_{args.random_state}{postfix}.csv"
    if (not os.path.exists(fn_1)) or (not os.path.exists(fn_2)):
        df_1, df_2 = embed_dfs(embedding_model=args.embedding_model, 
                           step_size=args.step_size, 
                           random_state=args.random_state,
                           use_spurious_correction=args.use_spurious_correction)
        df_1.to_csv(fn_1)
        df_2.to_csv(fn_2)
        
    df_1 = pd.read_csv(fn_1)
    df_2 = pd.read_csv(fn_2)
    runs = {
        "Synthetic": {
            "All": df_1,
            "Clicked": df_1[df_1['clicked_links']>0].copy(),
        },
        "Natural": {
            "All": df_2,
            "Clicked": df_2[df_2['clicked_links']>0].copy(),
        },
        "Combined": {
            "Clicked": pd.concat([df_1[df_1['clicked_links']>0].copy(), 
                                  df_2[df_2['clicked_links']>0].copy()]),
        }
    }
    
    for corpus in runs:
        for behavior in runs[corpus]:
            df = runs[corpus][behavior]
            df = df[~df['nDCV'].isna()].copy()
            X = df[['prior',
                    'nDCVs', 
                   ]].values
            X = sm.add_constant(X)
            y = df['posterior'].values
            model = sm.OLS(y, X)
            results = model.fit()
            print(f"{corpus} & {behavior} & {int(results.nobs)} & {results.params[-1]:.3f} & {results.pvalues[-1]:.3f} & {results.rsquared:.3f}\\\\")    

main()