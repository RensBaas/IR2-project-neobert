import itertools, json, torch, random, os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm

class Duo(object):
    
    def __init__(self, embedding_model='all-MiniLM-L6-v2', step_size=1, random_state=None):
        self.pca = PCA(n_components=1)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embeddings = dict()
        self.docs = dict()
        self.Z = dict() # for Z-score normalization
        self.step_size = step_size
        if random_state:
            self.set_random_state(random_state)
        
    def set_random_state(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def embed_texts(self, texts, normalize=False):
        e = self.embedding_model.encode(texts)
        if normalize:
            return e / np.linalg.norm(e, axis=1, keepdims=True) # l2 normalized
        return e
           
    def embed(self, transform_docs, fit_docs=None, spurious_docs=[], fixed_Z=None):
        self.docs = transform_docs
        ids = list(self.docs.keys())
        texts = [self.docs[_id]['text'] for _id in ids]

        spurious_embeddings_1 = None
        spurious_embeddings_2 = None
        if len(spurious_docs)==2: # remove spurious correlations
            spurious_embeddings_1 = self.embed_texts([spurious_docs[0][_id]['text'] for _id in spurious_docs[0]], normalize=True)
            spurious_embeddings_2 = self.embed_texts([spurious_docs[1][_id]['text'] for _id in spurious_docs[1]], normalize=True)
            subspace = (spurious_embeddings_1.mean(axis=0) - spurious_embeddings_2.mean(axis=0))
            
        if fit_docs:
            fit_texts = [fit_docs[_id]['text'] for _id in fit_docs]
            fit_embeddings = self.embed_texts(fit_texts)
            if len(spurious_docs)==2:
                fit_embeddings = fit_embeddings - np.multiply(np.tile(subspace, (fit_embeddings.shape[0], 1)).T, 
                                                                         np.dot(fit_embeddings, subspace)).T
            self.pca.fit(fit_embeddings)
        
        embeddings_raw = self.embed_texts(texts)
        
        if len(spurious_docs)==2: # remove spurious correlations
            subspace = (spurious_embeddings_1.mean(axis=0) - spurious_embeddings_2.mean(axis=0))
            embeddings_raw = embeddings_raw - np.multiply(np.tile(subspace, (embeddings_raw.shape[0], 1)).T, 
                                                                     np.dot(embeddings_raw, subspace)).T
            
        embeddings_transformed = self.pca.transform(embeddings_raw).flatten()
        
        self.embeddings = dict() # clear
        for i, _id in enumerate(ids):
            self.embeddings[_id] = embeddings_transformed[i]
         
        if fixed_Z:
            self.Z = fixed_Z
        elif len(ids) > 8:
            self.update_Z(sample_n=40320)
        else: 
            self.update_Z()
            
    def N(self):
        return len(self.embeddings.keys())
    
    def sign(self, arr):
        """Returns +1 if the row is dominated by positive numbers and 
        -1 if the row is dominated by negative numbers"""
        gt = (arr>0).astype(int)
        if (len(arr.shape)>1):
            return 2.0*(((gt.sum(axis=1))/arr.shape[1])>0.5).astype(int)-1.0
        elif len(arr)==1:
            return 2.0*(gt[0]).astype(int)-1.0
    
    def metric(self, batch, use_sign=False):
        if use_sign:
            return batch.var(axis=1)*self.sign(batch)
        return batch.var(axis=1)
    
    def normalized_metric(self, batch, top_k, use_sign=False):
        assert top_k in self.Z
        return (self.metric(batch, use_sign) -  self.Z[top_k]['mean']) / self.Z[top_k]['std']
        
    def Duo_raw_batch(self, batch, use_sign=False, count_full_ranking=False): # normalized discounted cumulative variance
        # batch should be a 2 dimensional array of retrieval-ranked document embeddings
        cumulative = np.zeros(batch.shape[0])
        for k in range(self.step_size, self.N()+int(count_full_ranking), self.step_size):
            if k>1:
                cumulative += ( self.metric(batch[:, :k], use_sign) / np.log2(k))
        return cumulative
    
    def Duo_batch_(self, batch, normalization="max", use_sign=False):
        sign = 1.0
        if use_sign:
            out = self.Duo_raw_batch(batch, use_sign)
            sign = self.sign( out )
        if normalization=='Z':
            return sign*(1.0 - ((self.Duo_raw_batch(batch) -  self.Z['mean']) / self.Z['std']))
        if not (self.Z['max'] - self.Z['min']):
            return sign*((0.0*self.Duo_raw_batch(batch))+0.5)
        return 1.0 - ((self.Duo_raw_batch(batch) -  self.Z['min']) / (self.Z['max'] - self.Z['min']) )
    
    def Duo_batch(self, rankings, normalization="max", use_sign=False):
        # ranking should be a 2d list, where each element is an ordered list of document IDs
        instance = np.array([[self.embeddings[_id] for _id in ranking] for ranking in rankings])
        return self.Duo_batch_(instance, normalization, use_sign)
    
    def Duo(self, ranking):
        return self.Duo_batch([ranking])[0]
    
    def permutations(self, lst, k, sample_n=0):
        if sample_n:
            p = np.random.permutation(k)
            yield tuple(np.array(lst)[p])
        else:
            p = itertools.permutations(lst, k)
            for permutation in p:
                yield permutation
                
    def yield_batch_permutations(self, a, batch_size=40320, sample_with_replacement=False):
        permutations = []
        for i, p in enumerate(itertools.permutations(a)):
            if sample_with_replacement:
                random_idx = np.random.permutation(len(a))
                permutations.append(list(np.array(a)[random_idx]))
            else:
                permutations.append(list(p))
            if (i+1)%(batch_size)==0:
                _permutations = permutations
                permutations = []
                yield np.array(_permutations)
        if len(permutations):
            yield np.array(permutations)
      
    def update_Z(self, sample_n=0, batch_size=40320):     #40320
        X_sum = 0
        X_2_sum = 0
        X_min = 1e10
        X_max = -1e10
        argmax = None
        argmin = None
        n = 0

        emb = list(self.embeddings.values())
        inv_embeddings = {self.embeddings[key]: key for key in self.embeddings}
        for i, p in enumerate(self.yield_batch_permutations(emb, batch_size=batch_size, sample_with_replacement=bool(sample_n))):

            if sample_n and n>=sample_n:
                break

            metric = self.Duo_raw_batch(p)
            X_sum += metric.sum()
            X_2_sum += (metric**2).sum()
            if metric.max() > X_max:
                X_max = metric.max()
                argmax = [inv_embeddings[x] for x in p[metric.argmax(), :]]
            if metric.min() < X_min:
                X_min = metric.min()
                argmin = [inv_embeddings[x] for x in p[metric.argmin(), :]]
            n += len(p)
              
        mean = X_sum/float(n)
        mean_2 = X_2_sum/float(n)
        self.Z = {
            'mean': mean,
            'std': np.sqrt( mean_2 - (mean**2) ),
            'max': X_max,
            'min': X_min,
            'argmax': argmax,
            'argmin': argmin
        }
    
def get_relevant_ranking(retrieved, query_idx, qrels, qrel_threshold=4):
    ranking = sorted(retrieved[query_idx], key=retrieved[query_idx].get, reverse=True)
    ranking_relevant_subset = [ doc_id for doc_id in ranking if ( (doc_id in qrels[query_idx]) and (qrels[query_idx][doc_id]>=qrel_threshold) ) ]
    return ranking_relevant_subset

def get_corpus_subset(corpus, doc_ids=[]):
    return {doc_id: corpus[doc_id] for doc_id in corpus if doc_id in doc_ids}

def get_relevant_corpus_retrieved(corpus, retrieved, query_idx, qrels, qrel_threshold=4):
    return get_corpus_subset(
        corpus,
        doc_ids=get_relevant_ranking(retrieved, query_idx, qrels, qrel_threshold)
    )

def get_relevant_corpus(corpus, query_idx, qrels, qrel_threshold=4):
    return {doc_id: corpus[doc_id] for doc_id in corpus if ( (doc_id in qrels[query_idx]) and (qrels[query_idx][doc_id]>=qrel_threshold) )}
            
def indexical_bias_results(retrieved, corpus, qrels, qrel_threshold=4, step_size=2, fit_corpus=None, fit_qrels=None, spurious_corpus=None, path_to_Z_score=None, embedding_model='all-MiniLM-L6-v2', random_state=7):

    fixed_Z = None
    if path_to_Z_score:
        with open(path_to_Z_score, 'r') as infile:
            fixed_Z = json.load(infile)
    
    results = []
    d = Duo(embedding_model=embedding_model, step_size=step_size, random_state=random_state)
    print('length of retrieved', len(retrieved.keys()))
    for query_idx in tqdm(retrieved):
        if query_idx not in qrels:
            continue
            
        success = True
        corpus_relevant_subset = get_relevant_corpus_retrieved(corpus, retrieved, query_idx, qrels, qrel_threshold)
        fit_corpus_relevant_subset = corpus_relevant_subset
        
        if not len(corpus_relevant_subset):
            results.append(None)
            continue # can't operate on an empty set
        
        # if there is an alternative corpus to fit embeddings to
        if fit_corpus and fit_qrels:
            fit_corpus_relevant_subset = get_relevant_corpus(fit_corpus, query_idx, fit_qrels, qrel_threshold)
            
        spurious_docs = []
        if spurious_corpus:
            spurious_docs_1 = [ doc_id for doc_id in spurious_corpus if query_idx+'_p1' in doc_id]
            spurious_docs_1 = {doc_id: spurious_corpus[doc_id] for doc_id in spurious_corpus if doc_id in spurious_docs_1}
            spurious_docs_2 = [ doc_id for doc_id in spurious_corpus if query_idx+'_p2' in doc_id]
            spurious_docs_2 = {doc_id: spurious_corpus[doc_id] for doc_id in spurious_corpus if doc_id in spurious_docs_2}
            if len(spurious_docs_1) and len(spurious_docs_2):
                spurious_docs = [spurious_docs_1, spurious_docs_2]
            else:
                print("failed to remove spurious correlations for", query_idx)
                success = False
            
        if fixed_Z:
            d.embed(transform_docs=corpus_relevant_subset, 
                    fit_docs=fit_corpus_relevant_subset,
                    spurious_docs=spurious_docs,
                    fixed_Z=fixed_Z[query_idx])
        else:
            d.embed(transform_docs=corpus_relevant_subset, 
                    fit_docs=fit_corpus_relevant_subset,
                    spurious_docs=spurious_docs,
                   )
        nDuo_score = d.Duo(ranking=get_relevant_ranking(retrieved, query_idx, qrels))
        if (type(nDuo_score) in {float, int}) or (isinstance(nDuo_score, np.floating) and np.isfinite(nDuo_score)):
            if success:
                results.append(nDuo_score)
            else:
                results.append(None)
        else:
            results.append(None)
            print(type(nDuo_score))
            
    print('length of results', len(results))
    return np.array(results)