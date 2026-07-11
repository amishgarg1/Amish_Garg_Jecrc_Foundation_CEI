import string
import numpy as np
from typing import List, Dict, Any, Optional

class LocalVectorStore:
    """
    A lightweight, in-memory vector store that uses numpy for cosine similarity.
    Does not require any external databases (like chromadb) or native C++ compilers.
    """
    def __init__(self):
        self.embeddings = [] # List of np.ndarray
        self.documents = []  # List of Dict (matching text + metadata)

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Appends documents and their embeddings.
        """
        for doc, emb in zip(documents, embeddings):
            self.documents.append(doc)
            self.embeddings.append(np.array(emb, dtype=np.float32))

    def similarity_search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs Cosine Similarity search using numpy.
        """
        if not self.embeddings or not self.documents:
            return []

        emb_matrix = np.vstack(self.embeddings) # shape: (N, D)
        q_vec = np.array(query_embedding, dtype=np.float32) # shape: (D,)

        # Cosine similarity: A.B / (||A|| * ||B||)
        dot_products = np.dot(emb_matrix, q_vec)
        matrix_norms = np.linalg.norm(emb_matrix, axis=1)
        q_norm = np.linalg.norm(q_vec)

        # Avoid division by zero
        norms = matrix_norms * q_norm
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = dot_products / norms

        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["score"] = float(similarities[idx])
            doc["index"] = int(idx)
            results.append(doc)
        return results

    def clear(self):
        self.embeddings = []
        self.documents = []


class LocalEmbeddingModel:
    """
    Generates local embeddings using SentenceTransformer from HuggingFace.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._lazy_load()
        embs = self._model.encode(texts, show_progress_bar=False)
        return embs.tolist()

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        self._lazy_load()
        embs = self._model.encode(texts, show_progress_bar=False)
        return embs.tolist()


class HybridRetriever:
    """
    Combines vector search and BM25 keyword search, then applies optional Cross-Encoder re-ranking.
    """
    def __init__(self, vector_store: LocalVectorStore, embedding_model: LocalEmbeddingModel, documents: List[Dict[str, Any]]):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.documents = documents
        
        self.bm25 = None
        self._initialize_bm25()
        self._cross_encoder = None

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase, remove punctuation, split by space
        translator = str.maketrans("", "", string.punctuation)
        text_clean = text.lower().translate(translator)
        return text_clean.split()

    def _initialize_bm25(self):
        if not self.documents:
            return
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            print(f"Error initializing BM25: {e}")

    def update_documents(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self._initialize_bm25()

    def dense_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_emb = self.embedding_model.embed_queries([query])[0]
        return self.vector_store.similarity_search(query_emb, top_k=top_k)

    def sparse_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.bm25 is None or not self.documents:
            return []
        
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["score"] = float(bm25_scores[idx])
            doc["index"] = int(idx)
            results.append(doc)
        return results

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Combines dense and sparse search.
        alpha = 1.0 -> 100% dense (vector)
        alpha = 0.0 -> 100% sparse (BM25)
        """
        if alpha >= 1.0:
            return self.dense_search(query, top_k=top_k)
        if alpha <= 0.0:
            return self.sparse_search(query, top_k=top_k)

        # Retrieve more candidates from each, then merge and rank
        candidate_k = max(top_k * 2, 20)
        dense_results = self.dense_search(query, top_k=candidate_k)
        sparse_results = self.sparse_search(query, top_k=candidate_k)

        if not dense_results and not sparse_results:
            return []

        # Map document identifiers to scores
        dense_scores = {doc["text"]: doc["score"] for doc in dense_results}
        sparse_scores = {doc["text"]: doc["score"] for doc in sparse_results}

        # Normalize scores to [0, 1] range to make them comparable
        def min_max_normalize(scores_dict: Dict[str, float]) -> Dict[str, float]:
            if not scores_dict:
                return {}
            vals = list(scores_dict.values())
            min_v, max_v = min(vals), max(vals)
            diff = max_v - min_v
            if diff == 0:
                return {k: 1.0 for k in scores_dict}
            return {k: (v - min_v) / diff for k, v in scores_dict.items()}

        norm_dense = min_max_normalize(dense_scores)
        norm_sparse = min_max_normalize(sparse_scores)

        # Merge and calculate hybrid scores
        all_unique_texts = set(dense_scores.keys()).union(set(sparse_scores.keys()))
        merged_results = []

        # Find documents
        text_to_doc = {}
        for doc in dense_results + sparse_results:
            text_to_doc[doc["text"]] = doc

        for text in all_unique_texts:
            d_score = norm_dense.get(text, 0.0)
            s_score = norm_sparse.get(text, 0.0)
            hybrid_score = alpha * d_score + (1.0 - alpha) * s_score

            doc = text_to_doc[text].copy()
            doc["score"] = hybrid_score
            doc["dense_score"] = dense_scores.get(text, 0.0)
            doc["sparse_score"] = sparse_scores.get(text, 0.0)
            merged_results.append(doc)

        # Sort by hybrid score descending
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        return merged_results[:top_k]

    def re_rank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> List[Dict[str, Any]]:
        """
        Re-ranks retrieved documents using a Cross-Encoder.
        """
        if not documents:
            return []

        if self._cross_encoder is None:
            print(f"Loading cross-encoder: {model_name}...")
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(model_name)

        pairs = [[query, doc["text"]] for doc in documents]
        scores = self._cross_encoder.predict(pairs)

        re_ranked = []
        for doc, score in zip(documents, scores):
            r_doc = doc.copy()
            r_doc["re_rank_score"] = float(score)
            re_ranked.append(r_doc)

        re_ranked.sort(key=lambda x: x["re_rank_score"], reverse=True)
        return re_ranked[:top_k]


if __name__ == "__main__":
    print("Retrieval module ready.")
