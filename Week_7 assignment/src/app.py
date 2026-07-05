import streamlit as st
import os
import json
import time
from typing import List, Dict, Any

from ingestion import extract_text_from_pdf, extract_text_from_txt, load_hf_open_ragbench, extract_text_via_gemini
from chunking import chunk_document
from embeddings import get_embedding_model
from vector_store import get_vector_store
from retrieval import HybridRetriever
from generation import get_llm_client, build_prompt
from evaluate import run_evaluation

# Set page configuration with a premium dark theme feel
st.set_page_config(
    page_title="Grounded DocQA - Premium RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics (glassmorphism, gradients, modern typography, hover effects)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Header Gradient - White & Green Premium Theme */
    .header-container {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #d1fae5;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.05);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .header-title {
        background: linear-gradient(90deg, #047857 0%, #10b981 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #4b5563;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(16, 185, 129, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.03);
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 185, 129, 0.45);
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.10);
    }
    
    /* Metric Cards */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #059669, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Source Chunk Box */
    .source-chunk {
        background: #fcfdfd;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .source-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #059669;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Custom Sidebar Styles */
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
    
    /* Chat Bubbles */
    .chat-bubble-user {
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 1rem;
        border-radius: 12px 12px 0 12px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        align-self: flex-end;
    }
    
    .chat-bubble-assistant {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
        backdrop-filter: blur(8px);
        color: #1e293b;
        padding: 1.2rem;
        border-radius: 12px 12px 12px 0;
        margin-bottom: 1rem;
        border: 1px solid #a7f3d0;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35);
        background: linear-gradient(90deg, #059669 0%, #047857 100%);
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header-container">
    <div class="header-title">Grounded DocQA System</div>
    <div class="header-subtitle">Advanced RAG Pipeline with Hybrid Search, Cross-Encoder Re-ranking, and Groundedness Evaluation</div>
</div>
""", unsafe_allow_html=True)

# Initialize Session States
if "loaded_doc_text" not in st.session_state:
    st.session_state.loaded_doc_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "source_file_name" not in st.session_state:
    st.session_state.source_file_name = ""
if "open_ragbench_samples" not in st.session_state:
    st.session_state.open_ragbench_samples = []

# ==================== SIDEBAR CONFIGURATIONS ====================
st.sidebar.markdown("### ⚙️ Pipeline Control Panel")

# 1. API Keys & Engine Settings
st.sidebar.markdown("#### 1. Engine Mode")
engine_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Local (Offline & Free)", "Cloud API Mode"],
    help="Local uses sentence-transformers and local Python models. Cloud API connects to Gemini or Cohere."
)

api_key_gemini = ""
api_key_cohere = ""
api_key_pinecone = ""

if engine_mode == "Cloud API Mode":
    st.sidebar.markdown("##### 🔑 API Configuration")
    api_key_gemini = st.sidebar.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    api_key_cohere = st.sidebar.text_input("Cohere API Key", type="password", value=os.environ.get("COHERE_API_KEY", ""))
    
    use_pinecone = st.sidebar.checkbox("Use Pinecone Database")
    if use_pinecone:
        api_key_pinecone = st.sidebar.text_input("Pinecone API Key", type="password")
        pinecone_index = st.sidebar.text_input("Pinecone Index Name", value="rag-index")
else:
    use_pinecone = False

# 2. Embedding Model selection
st.sidebar.markdown("#### 2. Embedding Configuration")
if engine_mode == "Local (Offline & Free)":
    emb_option = "Local (all-MiniLM-L6-v2)"
    emb_type = "local"
    emb_model_name = "all-MiniLM-L6-v2"
else:
    emb_option = st.sidebar.selectbox("Embedding Engine", ["Gemini (text-embedding-004)", "Cohere (embed-english-v3.0)", "Local (all-MiniLM-L6-v2)"])
    if "Gemini" in emb_option:
        emb_type = "gemini"
        emb_model_name = "models/text-embedding-004"
    elif "Cohere" in emb_option:
        emb_type = "cohere"
        emb_model_name = "embed-english-v3.0"
    else:
        emb_type = "local"
        emb_model_name = "all-MiniLM-L6-v2"

# 3. LLM Client configuration
st.sidebar.markdown("#### 3. Generator Model")
if engine_mode == "Local (Offline & Free)":
    llm_type = "mock"
    llm_model_name = "Mock/Grounded-Extractor"
else:
    llm_option = st.sidebar.selectbox("LLM Provider", ["Gemini (gemini-1.5-flash)", "Cohere (command-r)", "Mock Offline Fallback"])
    if "Gemini" in llm_option:
        llm_type = "gemini"
        llm_model_name = "gemini-1.5-flash"
    elif "Cohere" in llm_option:
        llm_type = "cohere"
        llm_model_name = "command-r"
    else:
        llm_type = "mock"
        llm_model_name = "Mock/Grounded-Extractor"

# 4. Chunk Borders Settings
st.sidebar.markdown("#### 4. Chunk Settings")
chunk_size = st.sidebar.slider("Chunk Size (characters)", min_value=100, max_value=2000, value=500, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=50, step=10)

# 5. Retrieval Configuration
st.sidebar.markdown("#### 5. Search Optimization")
top_k = st.sidebar.slider("Retrieval Count (Top K Chunks)", min_value=1, max_value=15, value=5)
hybrid_alpha = st.sidebar.slider(
    "Hybrid Weight (Alpha)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.1,
    help="0.0 = BM25 Keyword Search only. 1.0 = Dense Vector Search only. 0.5 = Equal balanced fusion."
)
use_reranking = st.sidebar.checkbox(
    "Apply Cross-Encoder Re-ranking", 
    value=True,
    help="Uses a Cross-Encoder model to re-evaluate the relevance of retrieved chunks relative to the query."
)

# 6. Ingestion Panel
st.sidebar.markdown("#### 📥 Document Ingestion Module")
ingest_source = st.sidebar.radio("Data Source", ["Upload Custom File", "HF Open RAGBench Sample"])

# Action to process the documents
def process_text_content(text: str, name: str):
    with st.spinner("Processing document: splitting text, generating embeddings, and building vector store..."):
        # Chunking
        chunks = chunk_document(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Load Embedding Model
        emb_api_key = api_key_gemini if emb_type == "gemini" else (api_key_cohere if emb_type == "cohere" else None)
        try:
            emb_model = get_embedding_model(emb_type, api_key=emb_api_key, model_name=emb_model_name)
        except Exception as e:
            st.error(f"Error loading embedding model: {e}")
            return
            
        # Generate Embeddings
        chunk_texts = [c["text"] for c in chunks]
        try:
            chunk_embs = emb_model.embed_documents(chunk_texts)
        except Exception as e:
            st.error(f"Error generating embeddings: {e}. Check API key configuration.")
            return

        # Vector Store Creation
        if use_pinecone:
            try:
                vector_store = get_vector_store("pinecone", api_key=api_key_pinecone, index_name=pinecone_index, dimension=emb_model.dimension)
            except Exception as e:
                st.error(f"Pinecone Error: {e}")
                return
        else:
            vector_store = get_vector_store("local", persist_path="data/streamlit_vs")
            vector_store.clear()
            
        vector_store.add_documents(chunks, chunk_embs)
        
        # Retriever initialization
        retriever = HybridRetriever(vector_store, emb_model, chunks)
        
        # Update session states
        st.session_state.loaded_doc_text = text
        st.session_state.chunks = chunks
        st.session_state.embedding_model = emb_model
        st.session_state.vector_store = vector_store
        st.session_state.retriever = retriever
        st.session_state.source_file_name = name
        st.session_state.chat_history = [] # reset chat
        
        st.success(f"Successfully ingested '{name}' ({len(chunks)} chunks, embedding dimension: {emb_model.dimension})!")

if ingest_source == "Upload Custom File":
    uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file is not None:
        file_name = uploaded_file.name
        # Check if already processed
        if st.session_state.source_file_name != file_name:
            file_bytes = uploaded_file.read()
            try:
                if file_name.endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(file_bytes)
                else:
                    extracted_text = extract_text_from_txt(file_bytes)
                
                # Check for OCR/multimodal fallback if standard extraction fails
                if not extracted_text.strip() and file_name.endswith(".pdf"):
                    if api_key_gemini:
                        with st.spinner("No embedded text found. Attempting AI-driven OCR extraction via Gemini..."):
                            extracted_text = extract_text_via_gemini(file_bytes, api_key_gemini)
                    else:
                        st.sidebar.warning("⚠️ No text could be extracted from this PDF. It appears to be scanned or image-only. To extract text automatically using Gemini OCR, please provide a Gemini API Key in the 'Cloud API Mode' configuration.")
                
                if extracted_text.strip():
                    process_text_content(extracted_text, file_name)
                elif file_name.endswith(".pdf") and not api_key_gemini:
                    pass # Warning already shown above
                else:
                    st.sidebar.error("Could not extract any text from the file. It might be empty, password-protected, or unsupported.")
            except Exception as e:
                st.sidebar.error(f"Error parsing file: {e}")
else:
    # Load 5 samples from HF Dataset once
    if not st.session_state.open_ragbench_samples:
        with st.spinner("Fetching sample archives from Hugging Face Open RAGBench dataset..."):
            st.session_state.open_ragbench_samples = load_hf_open_ragbench(limit=5)
            
    if st.session_state.open_ragbench_samples:
        sample_questions = [f"Sample {i+1}: {s['question'][:60]}..." for i, s in enumerate(st.session_state.open_ragbench_samples)]
        selected_sample_idx = st.sidebar.selectbox("Choose HF Sample", range(len(sample_questions)), format_func=lambda x: sample_questions[x])
        
        if st.sidebar.button("Ingest Selected HF Sample"):
            sample = st.session_state.open_ragbench_samples[selected_sample_idx]
            # Context becomes our document
            process_text_content(sample["context"], f"HF Open RAGBench Sample (ID: {sample['id']})")
    else:
        st.sidebar.warning("Could not fetch Hugging Face dataset. Check internet connection.")


# ==================== MAIN PANEL TABS ====================
tab_chat, tab_chunks, tab_bench = st.tabs(["💬 Chat QA Dashboard", "📄 Documents & Chunk Explorer", "📊 System Benchmark & Validation"])

# --- TAB 1: CHAT QA ---
with tab_chat:
    if not st.session_state.loaded_doc_text:
        st.info("👈 Please load a document via the sidebar ingestion panel to start asking questions.")
    else:
        st.markdown(f"**Currently Active Document**: `{st.session_state.source_file_name}`")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-bubble-user">
                    <b>You:</b><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bubble-assistant">
                    <b>Assistant ({llm_model_name}):</b><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show source chunks in expander
                if "sources" in msg and msg["sources"]:
                    with st.expander("🔍 Grounded Source Citations & Scores"):
                        for i, doc in enumerate(msg["sources"]):
                            st.markdown(f"""
                            <div class="source-chunk">
                                <div class="source-meta">
                                    <span>Chunk #{doc.get('chunk_id', i)}</span>
                                    <span>Retrieval Score: {doc.get('score', 0.0):.4f}</span>
                                </div>
                                {doc.get('text', '')}
                            </div>
                            """, unsafe_allow_html=True)
                            
        # User input query
        query = st.chat_input("Ask a question grounded in your document...")
        
        if query:
            # Display user query
            st.markdown(f"""
            <div class="chat-bubble-user">
                <b>You:</b><br>{query}
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Start retrieval and answer generation
            with st.spinner("Retrieving facts and generating grounded answer..."):
                retriever = st.session_state.retriever
                
                # 1. Retrieve
                t_retrieve_start = time.time()
                retrieved = retriever.hybrid_search(query, top_k=top_k, alpha=hybrid_alpha)
                
                # 2. Re-rank
                if use_reranking and retrieved:
                    retrieved = retriever.re_rank(query, retrieved, top_k=min(3, top_k))
                else:
                    retrieved = retrieved[:3]
                t_retrieval_sec = time.time() - t_retrieve_start
                
                # 3. Generate Answer
                prompt = build_prompt(query, retrieved)
                
                llm_api_key = api_key_gemini if llm_type == "gemini" else (api_key_cohere if llm_type == "cohere" else None)
                try:
                    llm_client = get_llm_client(llm_type, api_key=llm_api_key)
                    answer = llm_client.generate(prompt)
                except Exception as e:
                    answer = f"Error during generation: {e}. Check generator configurations and API keys."
                
            # Display assistant answer
            st.markdown(f"""
            <div class="chat-bubble-assistant">
                <b>Assistant ({llm_model_name}):</b><br>{answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Source expander for current response
            with st.expander("🔍 Grounded Source Citations & Scores (Current Query)"):
                st.markdown(f"**Retrieval Latency**: `{t_retrieval_sec:.4f} seconds`")
                for i, doc in enumerate(retrieved):
                    # Pick correct display score
                    score_label = "Re-rank Score" if use_reranking else "Hybrid Score"
                    score_val = doc.get("re_rank_score", doc.get("score", 0.0))
                    
                    st.markdown(f"""
                    <div class="source-chunk">
                        <div class="source-meta">
                            <span>Chunk #{doc.get('chunk_id', i)}</span>
                            <span>{score_label}: {score_val:.4f}</span>
                        </div>
                        {doc.get('text', '')}
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": retrieved
            })

# --- TAB 2: CHUNK EXPLORER ---
with tab_chunks:
    if not st.session_state.loaded_doc_text:
        st.info("👈 Please load a document via the sidebar ingestion panel to inspect its chunk profile.")
    else:
        st.markdown(f"### 📊 Chunking Profile for `{st.session_state.source_file_name}`")
        
        # Display document and chunk stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Total Characters</div>
                <div class="metric-value">{len(st.session_state.loaded_doc_text):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Total Chunks</div>
                <div class="metric-value">{len(st.session_state.chunks)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_words = int(sum(c["word_count"] for c in st.session_state.chunks)/len(st.session_state.chunks)) if st.session_state.chunks else 0
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Avg Chunk Words</div>
                <div class="metric-value">{avg_words}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Embedding Dimensions</div>
                <div class="metric-value">{st.session_state.embedding_model.dimension}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### 🗂️ Document Chunks List")
        for i, chunk in enumerate(st.session_state.chunks):
            with st.expander(f"Chunk #{i} (Length: {chunk['char_count']} chars, {chunk['word_count']} words)"):
                st.code(chunk["text"], language="text")

# --- TAB 3: BENCHMARK & EVALUATION ---
with tab_bench:
    st.markdown("### 📊 Automated RAG Pipeline Benchmark & Evaluation")
    st.markdown("""
    Evaluate the end-to-end question answering pipeline against the Hugging Face `vectara/open_ragbench` dataset.
    This module tests your configuration against multi-document QA pairs, scoring both **retrieval recall** and **generation groundedness**.
    """)
    
    col_bench_1, col_bench_2 = st.columns([1, 2])
    
    with col_bench_1:
        st.markdown("#### ⚙️ Evaluation Settings")
        eval_samples = st.slider("Samples to Evaluate", min_value=1, max_value=10, value=3)
        
        btn_run_eval = st.button("🚀 Run Evaluation Pipeline")
        
    with col_bench_2:
        if btn_run_eval:
            with st.spinner(f"Running automated evaluation on {eval_samples} samples from Open RAGBench..."):
                # Fetch keys
                emb_api_key = api_key_gemini if emb_type == "gemini" else (api_key_cohere if emb_type == "cohere" else None)
                llm_api_key = api_key_gemini if llm_type == "gemini" else (api_key_cohere if llm_type == "cohere" else None)
                
                # Execute evaluation
                metrics = run_evaluation(
                    num_samples=eval_samples,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_type=emb_type,
                    embedding_model_name=emb_model_name,
                    llm_type=llm_type,
                    llm_api_key=llm_api_key,
                    alpha=hybrid_alpha,
                    use_rerank=use_reranking
                )
                
                st.markdown("### 📈 Evaluation Summary Report")
                
                ecol1, ecol2, ecol3 = st.columns(3)
                with ecol1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Retrieval Recall@K</div>
                        <div class="metric-value">{metrics['averages']['mean_retrieval_recall']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with ecol2:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Groundedness Score</div>
                        <div class="metric-value">{metrics['averages']['mean_groundedness']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with ecol3:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Answer Overlap (F1)</div>
                        <div class="metric-value">{metrics['averages']['mean_answer_f1']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown(f"**Mean Execution Latency per sample**: `{metrics['averages']['mean_latency_sec']:.2f} seconds`")
                
                # Load detail logs from JSON
                try:
                    with open("logs/validation_logs.json", "r", encoding="utf-8") as f:
                        logs = json.load(f)
                    
                    st.markdown("#### 📋 Detailed Validation Logs")
                    for r in logs["results"]:
                        with st.expander(f"Question: {r['question']}"):
                            st.write(f"**Golden Answer**: {r['golden_answer']}")
                            st.write(f"**Generated Answer**: {r['generated_answer']}")
                            
                            st.markdown("**Sample Metrics**:")
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric("Recall", f"{r['metrics']['recall']*100:.1f}%")
                            sc2.metric("Groundedness", f"{r['metrics']['groundedness']*100:.1f}%")
                            sc3.metric("F1 Answer similarity", f"{r['metrics']['answer_f1']*100:.1f}%")
                except Exception as e:
                    st.error(f"Error loading logs: {e}")
        else:
            # Check if log file already exists from a CLI run
            if os.path.exists("logs/validation_logs.json"):
                with open("logs/validation_logs.json", "r", encoding="utf-8") as f:
                    logs = json.load(f)
                summary = logs["summary"]
                st.info("💡 Existing validation log loaded. Showing latest CLI evaluation results.")
                
                ecol1, ecol2, ecol3 = st.columns(3)
                with ecol1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Retrieval Recall@K</div>
                        <div class="metric-value">{summary['averages']['mean_retrieval_recall']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with ecol2:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Groundedness Score</div>
                        <div class="metric-value">{summary['averages']['mean_groundedness']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with ecol3:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">Answer Overlap (F1)</div>
                        <div class="metric-value">{summary['averages']['mean_answer_f1']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("💡 Click the button above to run the automated benchmarking pipeline.")
