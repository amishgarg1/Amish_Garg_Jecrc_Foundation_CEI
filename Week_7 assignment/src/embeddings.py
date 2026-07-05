import os
from typing import List, Union
import numpy as np

class EmbeddingModel:
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class LocalEmbeddingModel(EmbeddingModel):
    """
    Uses sentence-transformers to generate vector embeddings locally.
    Defaults to 'all-MiniLM-L6-v2' (384 dimensions).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384 if "MiniLM" in model_name else 768

    def _lazy_load(self):
        if self._model is None:
            print(f"Loading local embedding model: {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            # Update dimension based on loaded model
            self._dimension = self._model.get_sentence_embedding_dimension()

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        self._lazy_load()
        embeddings = self._model.encode(queries, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        self._lazy_load()
        embeddings = self._model.encode(documents, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        self._lazy_load()
        return self._dimension


class GeminiEmbeddingModel(EmbeddingModel):
    """
    Uses Google's generativeai API to generate vector embeddings.
    Defaults to 'models/text-embedding-004' (768 dimensions).
    """
    def __init__(self, api_key: str = None, model_name: str = "models/text-embedding-004"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self._dimension = 768
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("Gemini API key is not configured.")
        import google.generativeai as genai
        embeddings = []
        for query in queries:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            embeddings.append(result['embedding'])
        return embeddings

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("Gemini API key is not configured.")
        import google.generativeai as genai
        embeddings = []
        # Process in batches to prevent API rate limit issues
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            result = genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type="retrieval_document"
            )
            embeddings.extend(result['embedding'])
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class CohereEmbeddingModel(EmbeddingModel):
    """
    Uses Cohere's client to generate vector embeddings.
    Defaults to 'embed-english-v3.0' (1024 dimensions).
    """
    def __init__(self, api_key: str = None, model_name: str = "embed-english-v3.0"):
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.model_name = model_name
        self._client = None
        self._dimension = 1024

    def _lazy_load(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("Cohere API key is not configured.")
            import cohere
            self._client = cohere.Client(self.api_key)

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        self._lazy_load()
        response = self._client.embed(
            texts=queries,
            model=self.model_name,
            input_type="search_query"
        )
        return response.embeddings

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        self._lazy_load()
        response = self._client.embed(
            texts=documents,
            model=self.model_name,
            input_type="search_document"
        )
        return response.embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_model(model_type: str = "local", api_key: str = None, model_name: str = None) -> EmbeddingModel:
    """
    Factory function to retrieve embedding model instance.
    """
    m_type = model_type.lower()
    if m_type == "local":
        return LocalEmbeddingModel(model_name or "all-MiniLM-L6-v2")
    elif m_type == "gemini":
        return GeminiEmbeddingModel(api_key, model_name or "models/text-embedding-004")
    elif m_type == "cohere":
        return CohereEmbeddingModel(api_key, model_name or "embed-english-v3.0")
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")

if __name__ == "__main__":
    # Test local embedding model
    try:
        model = get_embedding_model("local")
        emb = model.embed_queries(["This is a test query"])
        print(f"Local embedding success. Dimension: {len(emb[0])}")
    except Exception as e:
        print(f"Local embedding test error (probably packages still installing): {e}")
