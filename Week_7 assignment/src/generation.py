import os
from typing import List, Dict, Any

class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiLLMClient(LLMClient):
    """
    Interfaces with Google's Gemini models for text generation.
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("Gemini API key is not configured.")
        import google.generativeai as genai
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text


class CohereLLMClient(LLMClient):
    """
    Interfaces with Cohere's Command models for text generation.
    """
    def __init__(self, api_key: str = None, model_name: str = "command-r"):
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.model_name = model_name
        self._client = None

    def _lazy_load(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("Cohere API key is not configured.")
            import cohere
            self._client = cohere.Client(self.api_key)

    def generate(self, prompt: str) -> str:
        self._lazy_load()
        # Using newer Cohere v2 chat interface or standard chat
        response = self._client.chat(
            message=prompt,
            model=self.model_name
        )
        return response.text


class MockLLMClient(LLMClient):
    """
    A smart local fallback that extracts the most contextually relevant sentence
    containing overlapping words with the query. Works completely offline.
    """
    def __init__(self, context: str = ""):
        self.context = context

    def set_context(self, context: str):
        self.context = context

    def generate(self, prompt: str) -> str:
        # If context is not set, extract it from prompt if possible
        # Look for standard marker
        context = self.context
        if not context and "[CONTEXT START]" in prompt and "[CONTEXT END]" in prompt:
            try:
                context = prompt.split("[CONTEXT START]")[1].split("[CONTEXT END]")[0].strip()
            except Exception:
                pass
        
        # Try to find the query from the prompt
        query = ""
        if "Question:" in prompt:
            try:
                query = prompt.split("Question:")[1].split("\n")[0].strip()
            except Exception:
                pass

        if not context:
            return "[Mock LLM] No context provided. Please load documents to answer."

        # Smart extraction: clean context first by removing metadata source lines
        import re
        cleaned_lines = []
        for line in context.splitlines():
            # Remove "--- Source X (Relevance Score: Y) ---" headers
            if not re.match(r'^---\s*Source\s+\d+.*---\s*$', line.strip()):
                cleaned_lines.append(line)
        cleaned_context = "\n".join(cleaned_lines)

        # Split context into candidate sentences/segments
        sentences = re.split(r'(?<=[.!?])\s+|\n+', cleaned_context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "[Mock LLM] Empty context."

        if not query:
            # Just return the first two sentences
            return f"[Mock LLM (No query detected)] " + " ".join(sentences[:2])

        # Tokenize query
        query_words = set(re.findall(r'\w+', query.lower()))
        # Remove common stopwords
        stopwords = {"what", "is", "the", "of", "and", "a", "to", "in", "for", "on", "with", "as", "at", "by", "an", "this", "that", "are", "it"}
        query_words = query_words - stopwords

        best_sentence = ""
        max_overlap_score = 0
        
        for s in sentences:
            s_lower = s.lower()
            # Calculate a score based on how many query words are found as substrings in this segment
            score = sum(1 for word in query_words if word in s_lower)
            if score > max_overlap_score:
                max_overlap_score = score
                best_sentence = s

        if max_overlap_score > 0:
            return f"[Grounded Local Answer] {best_sentence}"
        else:
            # Fallback to returning the first segment
            return f"[Grounded Local Answer] {sentences[0]}"


def get_llm_client(model_type: str = "mock", api_key: str = None, model_name: str = None) -> LLMClient:
    """
    Factory function to retrieve LLM client instance.
    """
    m_type = model_type.lower()
    if m_type == "mock":
        return MockLLMClient()
    elif m_type == "gemini":
        return GeminiLLMClient(api_key, model_name or "gemini-1.5-flash")
    elif m_type == "cohere":
        return CohereLLMClient(api_key, model_name or "command-r")
    else:
        raise ValueError(f"Unknown LLM client type: {model_type}")


def build_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Builds a structured prompt for the LLM using retrieved chunks.
    Ensures answers are grounded and limits hallucinations.
    """
    context_str = ""
    for idx, doc in enumerate(retrieved_chunks):
        context_str += f"\n--- Source {idx+1} (Relevance Score: {doc.get('score', doc.get('re_rank_score', 0.0)):.4f}) ---\n"
        context_str += doc["text"] + "\n"

    prompt = f"""You are an advanced AI assistant that answers questions based solely on the provided reference documents.

Instructions:
1. Answer the question relying ONLY on the clear facts mentioned in the context.
2. If the answer cannot be found or reasonably inferred from the context, state: "I cannot find the answer in the provided documents."
3. Do NOT make up any information, URLs, or facts.
4. Keep the answer concise and grounded.

[CONTEXT START]{context_str}[CONTEXT END]

Question: {query}
Answer:"""
    return prompt

if __name__ == "__main__":
    test_chunks = [{"text": "Gemini 1.5 Flash is a lightweight, fast, and cost-efficient model developed by Google."}]
    p = build_prompt("What is Gemini 1.5 Flash?", test_chunks)
    client = get_llm_client("mock")
    print(client.generate(p))
