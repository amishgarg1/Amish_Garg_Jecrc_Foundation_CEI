import argparse
import json
import time
import os
import re
import numpy as np
from typing import List, Dict, Any

from ingestion import load_hf_open_ragbench
from chunking import chunk_document
from embeddings import get_embedding_model
from vector_store import get_vector_store
from retrieval import HybridRetriever
from generation import get_llm_client, build_prompt

def calculate_word_overlap(text1: str, text2: str) -> float:
    """
    Calculates normalized word overlap (F1-score style) between two texts, ignoring stopwords.
    """
    stopwords = {"what", "is", "the", "of", "and", "a", "to", "in", "for", "on", "with", "as", "at", "by", "an", "this", "that", "are", "it", "i", "you", "he", "she", "they", "we"}
    
    words1 = set(re.findall(r'\w+', text1.lower())) - stopwords
    words2 = set(re.findall(r'\w+', text2.lower())) - stopwords
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    precision = len(intersection) / len(words1)
    recall = len(intersection) / len(words2)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_groundedness(generated_answer: str, context: str) -> float:
    """
    Evaluates if the generated answer is grounded in the retrieved context.
    Returns percentage of generated answer content words present in the context.
    """
    stopwords = {"what", "is", "the", "of", "and", "a", "to", "in", "for", "on", "with", "as", "at", "by", "an", "this", "that", "are", "it", "i", "cannot", "find", "answer", "provided", "documents", "relying", "only", "on", "facts"}
    
    gen_words = re.findall(r'\w+', generated_answer.lower())
    gen_content_words = [w for w in gen_words if w not in stopwords]
    
    if not gen_content_words:
        return 1.0 # If no content words (e.g. "I don't know"), it's safe (grounded)
        
    context_words = set(re.findall(r'\w+', context.lower()))
    
    grounded_count = sum(1 for w in gen_content_words if w in context_words)
    return grounded_count / len(gen_content_words)

def run_evaluation(num_samples: int = 5, chunk_size: int = 500, chunk_overlap: int = 50, 
                   embedding_type: str = "local", embedding_model_name: str = "all-MiniLM-L6-v2",
                   llm_type: str = "mock", llm_api_key: str = None, 
                   alpha: float = 0.5, use_rerank: bool = True) -> Dict[str, Any]:
    
    print("\n" + "="*50)
    print("STARTING RAG PIPELINE EVALUATION")
    print(f"Embedding Model: {embedding_type} ({embedding_model_name})")
    print(f"LLM Client: {llm_type}")
    print(f"Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")
    print(f"Hybrid Search Alpha: {alpha} (0=BM25, 1=Vector)")
    print(f"Re-ranking: {use_rerank}")
    print("="*50 + "\n")
    
    # 1. Load Dataset
    samples = load_hf_open_ragbench(limit=num_samples)
    if not samples:
        print("Failed to load dataset samples or dataset is empty. Using fallback synthetic evaluation samples.")
        samples = [
            {
                "id": 0,
                "question": "What is Open RAGBench?",
                "context": "Vectara Open RAGBench is a high-quality benchmark designed for evaluating Retrieval-Augmented Generation (RAG) systems. It features 1,000 arXiv PDF documents containing mixtures of text, tables, and images. It evaluates retrieval recall, context parsing, and response accuracy.",
                "answer": "A benchmark by Vectara for evaluating RAG systems on arXiv PDF documents."
            },
            {
                "id": 1,
                "question": "How many documents are in Open RAGBench?",
                "context": "The Open RAGBench dataset has 1,000 PDF papers distributed across arXiv categories. Over 3,000 question-answer pairs are included in this benchmark to evaluate retrieval and answer generation accuracy.",
                "answer": "1,000 PDF papers."
            },
            {
                "id": 2,
                "question": "What is the purpose of hybrid search in RAG?",
                "context": "Hybrid search in RAG combines sparse retrieval (like BM25, which matches exact keywords) with dense retrieval (like vector similarity, which matches semantic concepts). This combination leverages the precision of keywords and the conceptual breadth of vector search.",
                "answer": "To combine keyword matching (BM25) and semantic vector search."
            }
        ]
        num_samples = len(samples)

    # 2. Setup Models
    emb_model = get_embedding_model(embedding_type, model_name=embedding_model_name)
    llm_client = get_llm_client(llm_type, api_key=llm_api_key)
    
    eval_results = []
    
    recalls = []
    groundedness_scores = []
    answer_overlaps = []
    latencies = []
    
    for s_idx, sample in enumerate(samples):
        print(f"\nEvaluating Sample {s_idx+1}/{num_samples} (ID: {sample['id']})")
        print(f"Question: {sample['question']}")
        
        start_time = time.time()
        
        # A. Chunk Context
        doc_text = sample["context"]
        chunks = chunk_document(doc_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # B. Embed Chunks
        chunk_texts = [c["text"] for c in chunks]
        chunk_embs = emb_model.embed_documents(chunk_texts)
        
        # C. Store in Vector Database
        # Use a temporary persist path for local vector store
        vs_path = f"data/eval_vs_{s_idx}"
        vector_store = get_vector_store("local", persist_path=vs_path)
        vector_store.clear()
        vector_store.add_documents(chunks, chunk_embs)
        
        # D. Setup Retriever
        retriever = HybridRetriever(vector_store, emb_model, chunks)
        
        # E. Retrieve Chunks
        retrieved_chunks = retriever.hybrid_search(sample["question"], top_k=5, alpha=alpha)
        
        # F. Re-rank
        if use_rerank and retrieved_chunks:
            retrieved_chunks = retriever.re_rank(sample["question"], retrieved_chunks, top_k=3)
        else:
            retrieved_chunks = retrieved_chunks[:3]
            
        # G. Evaluate Retrieval (Recall@K)
        # Check if the text of the golden answer or key terms overlap with retrieved chunks
        gold_answer = sample["answer"]
        gold_terms = set(re.findall(r'\w+', gold_answer.lower())) - {"what", "is", "the", "of", "and", "a", "to", "in"}
        
        overlap_found = False
        retrieved_combined_text = " ".join([c["text"].lower() for c in retrieved_chunks])
        
        if gold_terms:
            overlap_pct = sum(1 for term in gold_terms if term in retrieved_combined_text) / len(gold_terms)
            overlap_found = overlap_pct > 0.4 # Consider retrieved if 40% of answer terms present
        else:
            overlap_found = True
            
        recall = 1.0 if overlap_found else 0.0
        recalls.append(recall)
        
        # H. Generate Answer
        prompt = build_prompt(sample["question"], retrieved_chunks)
        gen_answer = llm_client.generate(prompt)
        
        # I. Evaluate Generation
        groundedness = evaluate_groundedness(gen_answer, retrieved_combined_text)
        ans_similarity = calculate_word_overlap(gen_answer, gold_answer)
        
        groundedness_scores.append(groundedness)
        answer_overlaps.append(ans_similarity)
        
        elapsed = time.time() - start_time
        latencies.append(elapsed)
        
        print(f"Generated: {gen_answer}")
        print(f"Golden Answer: {gold_answer}")
        print(f"Recall: {recall} | Groundedness: {groundedness:.2f} | Answer F1: {ans_similarity:.2f} | Time: {elapsed:.2f}s")
        
        eval_results.append({
            "sample_id": sample["id"],
            "question": sample["question"],
            "retrieved_chunks": [{"id": c.get("chunk_id"), "score": c.get("score", c.get("re_rank_score", 0.0)), "text": c["text"]} for c in retrieved_chunks],
            "generated_answer": gen_answer,
            "golden_answer": gold_answer,
            "metrics": {
                "recall": recall,
                "groundedness": groundedness,
                "answer_f1": ans_similarity,
                "latency_sec": elapsed
            }
        })
        
        # Clean up vector store files
        vector_store.clear()
        
    # Calculate global averages
    metrics_summary = {
        "dataset_evaluated": "vectara/open_ragbench",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configurations": {
            "num_samples": num_samples,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_type": embedding_type,
            "embedding_model_name": embedding_model_name,
            "embedding_dimension": emb_model.dimension,
            "llm_type": llm_type,
            "hybrid_alpha": alpha,
            "re_ranking": use_rerank
        },
        "averages": {
            "mean_retrieval_recall": float(np.mean(recalls)),
            "mean_groundedness": float(np.mean(groundedness_scores)),
            "mean_answer_f1": float(np.mean(answer_overlaps)),
            "mean_latency_sec": float(np.mean(latencies))
        }
    }
    
    # Save validation logs
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/validation_logs.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": metrics_summary,
            "results": eval_results
        }, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*50)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {os.path.abspath(log_file)}")
    print(f"Mean Retrieval Recall: {metrics_summary['averages']['mean_retrieval_recall']:.2f}")
    print(f"Mean Groundedness Score: {metrics_summary['averages']['mean_groundedness']:.2f}")
    print(f"Mean Answer Word Overlap (F1): {metrics_summary['averages']['mean_answer_f1']:.2f}")
    print(f"Mean Latency: {metrics_summary['averages']['mean_latency_sec']:.2f}s")
    print("="*50 + "\n")
    
    return metrics_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to evaluate")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--embedding", type=str, default="local", help="Embedding type: local, gemini, cohere")
    parser.add_argument("--llm", type=str, default="mock", help="LLM type: mock, gemini, cohere")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid alpha weight (0=BM25, 1=Vector)")
    parser.add_argument("--no_rerank", action="store_true", help="Disable re-ranking")
    
    args = parser.parse_args()
    
    run_evaluation(
        num_samples=args.samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        embedding_type=args.embedding,
        llm_type=args.llm,
        alpha=args.alpha,
        use_rerank=not args.no_rerank
    )
