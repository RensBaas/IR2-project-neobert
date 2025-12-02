from datasets import load_dataset

def load_wiki_balance(subset='synthetic'):
    
    corpus = {}
    queries = {}
    qrels = {}
     
    data_corpus = load_dataset(f"SALT-NLP/wiki-balance-{subset}", "corpus")
    for line in data_corpus['corpus']:
        corpus[line['_id']] = {
            'text': line['text'],
            'title': line['title']
        }
        
    data_queries = load_dataset(f"SALT-NLP/wiki-balance-{subset}", "queries")
    for line in data_queries['queries']:
        queries[line['_id']] = line['text']
        
    data_qrels = load_dataset(f"SALT-NLP/wiki-balance-{subset}-qrels", "test")    
    for line in data_qrels['test']:
        query_id, corpus_id, score = line["query-id"], line["corpus-id"], line["score"]

        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
        
    return corpus, queries, qrels