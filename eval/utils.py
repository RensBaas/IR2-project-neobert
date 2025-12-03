import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Custom encoder with CLS pooling (same as original code)
class CLSBiEncoder:
    def __init__(self, model_path, trust_remote_code=False, batch_size=64):
        print(f"Initializing encoder for {model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code)
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