import sys
from src.utils import calculate_emi, check_eligibility, RecursiveCharacterTextSplitter
from src.retrieval import LocalVectorStore, LocalEmbeddingModel, HybridRetriever
from src.generation import MockLLMClient, build_prompt
from src.validation import validate_response_grounding, find_sentences_citations
from src.synthetic_data import LOAN_POLICIES

def run_tests():
    print("=" * 60)
    print("RUNNING AI LOAN ADVISORY CHATBOT VERIFICATION TESTS")
    print("=" * 60)
    
    # ------------------ TEST 1: MATH FORMULAS ------------------
    print("\n[TEST 1] Verifying EMI and Eligibility mathematical formulations...")
    emi_out = calculate_emi(1000000, 9.0, 120)
    print(f"Computed EMI: {emi_out['emi']} (Expected ~12667.58)")
    assert abs(emi_out["emi"] - 12667.58) < 1.0, "EMI calculation mismatch!"
    print("EMI check: PASSED")
    
    # FOIR ratio = 60%, so max EMI = 60000 * 0.60 - 10000 = 26000.
    # Desired Loan EMI for 30 Lakhs is ~26991.84.
    elig_out = check_eligibility(60000, 10000, 3000000, 9.0, 240)
    print(f"Eligible: {elig_out['eligible']}, Status: {elig_out['status']}, Max Allowed EMI: {elig_out['max_emi_allowed']}")
    assert not elig_out["eligible"], "Should not be eligible!"
    
    elig_out_2 = check_eligibility(60000, 10000, 2500000, 9.0, 240)
    print(f"Eligible for smaller loan: {elig_out_2['eligible']}, Status: {elig_out_2['status']}")
    assert elig_out_2["eligible"], "Should be eligible for smaller loan!"
    print("Eligibility check: PASSED")
    
    # ------------------ TEST 2: TEXT SPLITTER ------------------
    print("\n[TEST 2] Verifying RecursiveCharacterTextSplitter...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    test_text = "This is a sentence. This is another sentence. And here is a third paragraph splitter to check."
    chunks = splitter.split_text(test_text)
    print(f"Text split into {len(chunks)} chunks.")
    for idx, chunk in enumerate(chunks):
        print(f"  Chunk {idx+1}: '{chunk}' (len: {len(chunk)})")
        assert len(chunk) <= 100, f"Chunk {idx+1} exceeds maximum size!"
    print("Text splitter check: PASSED")
    
    # ------------------ TEST 3: VECTOR STORE & HYBRID RETRIEVAL ------------------
    print("\n[TEST 3] Verifying LocalVectorStore indexing and search...")
    vstore = LocalVectorStore()
    embedding_model = LocalEmbeddingModel()
    
    test_docs = LOAN_POLICIES[:2]
    chunks_to_index = []
    texts_to_embed = []
    
    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    for doc in test_docs:
        split_chunks = doc_splitter.split_text(doc["text"])
        for c_idx, text in enumerate(split_chunks):
            chunks_to_index.append({
                "id": f"{doc['id']}_chunk_{c_idx}",
                "category": doc["category"],
                "title": doc["title"],
                "text": text
            })
            texts_to_embed.append(text)
            
    print(f"Embedding and indexing {len(chunks_to_index)} chunks...")
    embeddings = embedding_model.embed_documents(texts_to_embed)
    vstore.add_documents(chunks_to_index, embeddings)
    
    retriever = HybridRetriever(vstore, embedding_model, chunks_to_index)
    
    query = "What interest rate is offered for home loans?"
    results = retriever.hybrid_search(query, top_k=2, alpha=0.5)
    print(f"Query: '{query}'")
    print(f"Retrieved {len(results)} results:")
    for idx, r in enumerate(results):
        print(f"  Result {idx+1} (Score: {r['score']:.4f}): Title: '{r['title']}' | Text: '{r['text'][:80]}...'")
    
    assert len(results) > 0, "No documents retrieved!"
    assert "home" in results[0]["text"].lower() or "interest" in results[0]["text"].lower(), "Retrieved irrelevant documents!"
    print("Vector Store & Retrieval check: PASSED")
    
    # ------------------ TEST 4: OFFLINE GENERATION ------------------
    print("\n[TEST 4] Verifying Offline Grounded LLM Client generation...")
    prompt = build_prompt(query, results)
    client = MockLLMClient()
    answer = client.generate(prompt)
    print(f"Generated Grounded Answer:\n{answer}")
    assert "[Grounded Local Answer]" in answer, "Answer is not correctly formatted by local client!"
    print("Offline generation check: PASSED")
    
    # ------------------ TEST 5: GROUNDING VALIDATION ------------------
    print("\n[TEST 5] Verifying Grounding validation and citation logic...")
    val_out = validate_response_grounding(answer, results)
    print(f"Validation status: {val_out['status']}, Grounding Score: {val_out['grounding_score']}")
    print(f"Unverified tokens: {val_out['unverified_tokens']}")
    assert val_out["status"] in ["Pass", "Warning"], "Should not completely fail validation!"
    
    hallucinated_answer = "Vanguard offers home loans at a fixed interest rate of 4.5% per annum for a 50-year tenure."
    val_out_bad = validate_response_grounding(hallucinated_answer, results)
    print(f"Bad validation status: {val_out_bad['status']}, Grounding Score: {val_out_bad['grounding_score']}")
    print(f"Bad unverified tokens: {val_out_bad['unverified_tokens']}")
    assert val_out_bad["status"] == "Fail", "Hallucinated answer should fail grounding check!"
    assert "4.5%" in val_out_bad["unverified_tokens"], "Unverified rate 4.5% not flagged!"
    
    citations = find_sentences_citations(answer, results)
    print("Citations matched:")
    for cit in citations:
        print(f"  Sentence: \"{cit['sentence_text']}\" -> Cited source: '{cit['cited_source_title']}' (Score: {cit['overlap_score']})")
    
    print("Grounding Validation & Citations checks: PASSED")
    
    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
