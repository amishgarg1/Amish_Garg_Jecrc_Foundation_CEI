import re
from typing import List, Dict, Any

def extract_key_tokens(text: str) -> List[str]:
    """
    Extracts key informational units: numbers, percentages, monetary amounts,
    and capitalized alphanumeric sequences (potential proper nouns, bank names).
    """
    tokens = set()
    
    # Extract percentages (e.g., 8.45%, 12%)
    percentages = re.findall(r'\b\d+(?:\.\d+)?%', text)
    tokens.update(percentages)
    
    # Extract numbers (with or without commas/decimals)
    numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', text)
    for num in numbers:
        if len(num) > 1 or num in ['0']:
            tokens.add(num)
            
    # Extract capital letter words (Proper nouns)
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    words = cleaned_text.split()
    for w in words:
        if w[0].isupper() and w.lower() not in {
            "i", "the", "a", "an", "and", "or", "but", "if", "then", "else", "at", "by", 
            "for", "from", "in", "into", "of", "off", "on", "onto", "out", "over", "to", 
            "up", "with", "is", "are", "was", "were", "be", "been", "being", "have", "has", 
            "had", "do", "does", "did", "not", "we", "you", "they", "he", "she", "it", "this"
        }:
            tokens.add(w)
            
    return sorted(list(tokens))

def validate_response_grounding(response: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates if the terms in the response exist within the retrieved source chunks.
    Calculates a grounding score based on how many key tokens in the response are verified in the context.
    """
    if not response or not retrieved_chunks:
        return {
            "grounding_score": 0.0,
            "unverified_tokens": [],
            "status": "Fail",
            "message": "No response or source context to validate."
        }
    
    # Clean system bracketed tags like [Grounded Local Answer] or [Mock LLM] from response
    cleaned_response = re.sub(r'^\[[^\]]+\]\s*', '', response)
    
    # Combine all source texts
    source_context = " ".join([chunk["text"] for chunk in retrieved_chunks]).lower()
    
    response_tokens = extract_key_tokens(cleaned_response)
    if not response_tokens:
        return {
            "grounding_score": 1.0,
            "unverified_tokens": [],
            "status": "Pass",
            "message": "No key numerical or proper noun tokens detected. Response is verified."
        }
    
    unverified = []
    verified_count = 0
    
    for token in response_tokens:
        token_lower = token.lower()
        if token_lower in source_context:
            verified_count += 1
        else:
            unverified.append(token)
            
    grounding_score = verified_count / len(response_tokens)
    
    status = "Pass"
    if grounding_score < 0.80:
        status = "Warning"
    if grounding_score < 0.50:
        status = "Fail"
        
    message = f"Grounding validation passed. Score: {grounding_score:.2f} ({verified_count}/{len(response_tokens)} key facts verified)."
    if status == "Warning":
        message = f"Grounding warning: Some terms in the response cannot be verified in the sources. Score: {grounding_score:.2f}."
    elif status == "Fail":
        message = f"Grounding check failed: High rate of unverified terms. Score: {grounding_score:.2f}."
        
    return {
        "grounding_score": round(grounding_score, 2),
        "unverified_tokens": unverified,
        "verified_tokens_count": verified_count,
        "total_tokens_count": len(response_tokens),
        "status": status,
        "message": message
    }

def find_sentences_citations(response: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Splits response into sentences and finds which source chunk (if any) contains a strong match.
    Uses basic token overlap/containment to link response sentences to source page numbers/names.
    """
    if not response:
        return []
        
    # Clean system bracketed tags like [Grounded Local Answer] or [Mock LLM] from response
    cleaned_response = re.sub(r'^\[[^\]]+\]\s*', '', response)
        
    # Split response into sentences
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_response.strip())
    citations = []
    
    for idx, sentence in enumerate(sentences):
        if len(sentence.strip()) < 10:
            continue
            
        best_match_chunk = None
        best_match_score = 0.0
        
        s_words = set(re.findall(r'\w+', sentence.lower()))
        stopwords = {"what", "is", "the", "of", "and", "a", "to", "in", "for", "on", "with", "as", "at", "by", "an", "this", "that", "are", "it", "will", "be", "has"}
        s_words = s_words - stopwords
        
        if not s_words:
            continue
            
        for chunk in retrieved_chunks:
            chunk_text_lower = chunk["text"].lower()
            match_count = sum(1 for word in s_words if word in chunk_text_lower)
            match_score = match_count / len(s_words)
            
            # Max score for exact match
            if sentence.lower().strip(" .!?") in chunk_text_lower:
                match_score = 1.0
                
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_chunk = chunk
                
        if best_match_score >= 0.35:
            citations.append({
                "sentence_index": idx,
                "sentence_text": sentence,
                "cited_source_title": best_match_chunk.get("title", "Document Section"),
                "cited_source_category": best_match_chunk.get("category", "General"),
                "cited_chunk_id": best_match_chunk.get("id", "chunk"),
                "overlap_score": round(best_match_score, 2)
            })
            
    return citations
