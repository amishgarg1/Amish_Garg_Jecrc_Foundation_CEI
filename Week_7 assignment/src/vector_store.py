import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class VectorStore:
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        raise NotImplementedError

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError


class LocalVectorStore(VectorStore):
    """
    A lightweight, zero-dependency NumPy-based vector store.
    Computes cosine similarity and supports saving/loading.
    """
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not documents:
            return
        
        # Ensure documents have text
        for doc in documents:
            if "text" not in doc:
                raise ValueError("Each document must have a 'text' field.")
        
        new_embeddings = np.array(embeddings, dtype=np.float32)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
            self.documents = list(documents)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.documents.extend(documents)
            
        if self.persist_path:
            self.save()

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.documents) == 0:
            return []
            
        q_emb = np.array(query_embedding, dtype=np.float32)
        
        # Compute cosine similarity
        # Cosine Sim = (A . B) / (||A|| * ||B||)
        # Normalize vectors first
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        norm_embeddings = self.embeddings / norms
        
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            q_norm = 1e-10
        norm_q = q_emb / q_norm
        
        similarities = np.dot(norm_embeddings, norm_q)
        
        # Get top-k indices sorted descending
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["score"] = float(similarities[idx])
            doc["index"] = int(idx)
            results.append(doc)
            
        return results

    def save(self) -> None:
        if not self.persist_path:
            return
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        
        # Save documents as JSON, embeddings as .npy
        docs_path = self.persist_path + ".json"
        embs_path = self.persist_path + ".npy"
        
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
            
        if self.embeddings is not None:
            np.save(embs_path, self.embeddings)

    def load(self) -> bool:
        if not self.persist_path:
            return False
        
        docs_path = self.persist_path + ".json"
        embs_path = self.persist_path + ".npy"
        
        if not os.path.exists(docs_path) or not os.path.exists(embs_path):
            return False
            
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            self.embeddings = np.load(embs_path)
            return True
        except Exception as e:
            print(f"Error loading local vector store: {e}")
            return False

    def clear(self) -> None:
        self.documents = []
        self.embeddings = None
        if self.persist_path:
            docs_path = self.persist_path + ".json"
            embs_path = self.persist_path + ".npy"
            if os.path.exists(docs_path):
                os.remove(docs_path)
            if os.path.exists(embs_path):
                os.remove(embs_path)


class PineconeVectorStore(VectorStore):
    """
    Vector store interface for Pinecone.
    """
    def __init__(self, api_key: str, index_name: str, dimension: int = 384):
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self._index = None
        self._initialize()

    def _initialize(self):
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}...")
            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self._index = pc.Index(self.index_name)

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        upsert_data = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = doc.get("id") or doc.get("chunk_id") or f"doc_{i}"
            doc_id = str(doc_id)
            # Store metadata
            metadata = {
                "text": doc.get("text", ""),
                "chunk_id": doc.get("chunk_id", i),
            }
            # Add any extra non-dict / simple metadata
            for k, v in doc.items():
                if k not in ["text", "chunk_id", "embeddings"] and isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
            
            upsert_data.append((doc_id, emb, metadata))
            
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(upsert_data), batch_size):
            self._index.upsert(vectors=upsert_data[i : i + batch_size])

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        response = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        results = []
        for i, match in enumerate(response.get("matches", [])):
            metadata = match.get("metadata", {})
            doc = {
                "text": metadata.get("text", ""),
                "chunk_id": metadata.get("chunk_id", i),
                "score": float(match.get("score", 0.0)),
                "id": match.get("id")
            }
            # Pull other metadata fields
            for k, v in metadata.items():
                if k not in ["text", "chunk_id"]:
                    doc[k] = v
            results.append(doc)
            
        return results


def get_vector_store(store_type: str = "local", **kwargs) -> VectorStore:
    """
    Factory function to retrieve vector store instance.
    """
    s_type = store_type.lower()
    if s_type == "local":
        persist_path = kwargs.get("persist_path", "data/vector_store")
        return LocalVectorStore(persist_path)
    elif s_type == "pinecone":
        api_key = kwargs.get("api_key")
        index_name = kwargs.get("index_name", "rag-index")
        dimension = kwargs.get("dimension", 384)
        if not api_key:
            raise ValueError("Pinecone API key is required.")
        return PineconeVectorStore(api_key, index_name, dimension)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")

if __name__ == "__main__":
    # Test local vector store
    store = get_vector_store("local", persist_path="data/test_vs")
    test_docs = [{"chunk_id": 0, "text": "Python is a programming language."}, {"chunk_id": 1, "text": "RAG helps ground LLM answers."}]
    test_embs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    store.add_documents(test_docs, test_embs)
    res = store.similarity_search([0.1, 0.2, 0.3], top_k=1)
    print("Local Vector Store search success:", res)
    store.clear()
