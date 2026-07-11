import os
from typing import List, Dict, Any

class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiLLMClient(LLMClient):
    """
    Interfaces with Google's Gemini models for text generation using google-generativeai.
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("Gemini API key is not configured. Please supply an API key in the sidebar.")
        import google.generativeai as genai
        # Configure dynamically with the user's current key
        genai.configure(api_key=self.api_key)
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
                raise ValueError("Cohere API key is not configured. Please supply an API key in the sidebar.")
            import cohere
            self._client = cohere.Client(self.api_key)

    def generate(self, prompt: str) -> str:
        self._lazy_load()
        response = self._client.chat(
            message=prompt,
            model=self.model_name
        )
        return response.text


class MockLLMClient(LLMClient):
    """
    A smart local fallback that extracts the most contextually relevant sentences
    containing overlapping words with the query. Works completely offline, ensuring data privacy.
    """
    def __init__(self, context: str = ""):
        self.context = context

    def set_context(self, context: str):
        self.context = context

    def generate(self, prompt: str) -> str:
        # Extract context and query from the prompt template if they are not pre-set
        context = self.context
        if not context and "[CONTEXT START]" in prompt and "[CONTEXT END]" in prompt:
            try:
                context = prompt.split("[CONTEXT START]")[1].split("[CONTEXT END]")[0].strip()
            except Exception:
                pass
        
        query = ""
        if "Question:" in prompt:
            try:
                query = prompt.split("Question:")[1].split("\n")[0].strip()
            except Exception:
                pass

        if not context:
            return "[Offline Private Advisor] No reference text found. Please upload documents or select a policy."

        # Clean metadata indicators from context
        import re
        cleaned_lines = []
        for line in context.splitlines():
            if not re.match(r'^---\s*Source\s+\d+.*---\s*$', line.strip()):
                cleaned_lines.append(line)
        cleaned_context = "\n".join(cleaned_lines)

        # Segment context into sentences
        sentences = re.split(r'(?<=[.!?])\s+|\n+', cleaned_context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 8]
        
        if not sentences:
            return "[Offline Private Advisor] No readable content extracted from references."

        if not query:
            return f"[Offline Private Advisor (No query detected)] Sources contain: " + " ".join(sentences[:2])

        # Tokenize query to check for matching overlaps
        query_words = set(re.findall(r'\w+', query.lower()))
        stopwords = {"what", "is", "the", "of", "and", "a", "to", "in", "for", "on", "with", "as", "at", "by", "an", "this", "that", "are", "it", "how", "does", "do", "any", "loan"}
        query_words = query_words - stopwords

        scored_sentences = []
        for s in sentences:
            s_lower = s.lower()
            score = sum(1.5 if word in s_lower else 0.0 for word in query_words)
            # Boost score slightly if exact phrases match
            for word in query_words:
                if f" {word} " in f" {s_lower} ":
                    score += 0.5
            if score > 0:
                scored_sentences.append((score, s))

        # Sort by relevance score descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if scored_sentences:
            # Pick the top matching sentence(s) to form a coherent answer
            top_matches = [s for score, s in scored_sentences[:3]]
            joined_matches = " ".join(top_matches)
            return f"[Grounded Local Answer] {joined_matches}"
        else:
            # Fallback to the first few sentences in the context
            return f"[Grounded Local Answer (Low Overlap)] " + " ".join(sentences[:2])


def get_llm_client(model_type: str = "mock", api_key: str = None, model_name: str = None) -> LLMClient:
    """
    Factory function to retrieve LLM client instance.
    """
    m_type = model_type.lower()
    if m_type == "mock" or m_type == "offline":
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
        title = doc.get("title", f"Document Chunk {idx+1}")
        category = doc.get("category", "General")
        score = doc.get("score", doc.get("re_rank_score", 0.0))
        context_str += f"\n--- Source {idx+1}: {title} [{category}] (Relevance Score: {score:.4f}) ---\n"
        context_str += doc["text"] + "\n"

    prompt = f"""You are an advanced AI loan advisory chatbot that answers questions based solely on the provided reference documents.

Instructions:
1. Answer the question relying ONLY on the clear facts mentioned in the context.
2. If the answer cannot be found or reasonably inferred from the context, state: "I cannot find the answer in the provided documents."
3. Do NOT make up any information, URLs, or facts.
4. Keep the answer concise, professional, and grounded.

[CONTEXT START]{context_str}[CONTEXT END]

Question: {query}
Answer:"""
    return prompt

if __name__ == "__main__":
    print("Generation module ready.")
