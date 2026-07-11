import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import os
from typing import List, Dict, Any

@st.cache_resource
def load_ml_model():
    """Loads the pre-trained loan approval classifier ML model."""
    pkl_path = "src/models/loan_classifier.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return None

from src.synthetic_data import LOAN_POLICIES
from src.utils import (
    extract_text_from_pdf,
    RecursiveCharacterTextSplitter,
    calculate_emi,
    check_eligibility
)
from src.retrieval import LocalVectorStore, LocalEmbeddingModel, HybridRetriever
from src.generation import get_llm_client, build_prompt
from src.validation import validate_response_grounding, find_sentences_citations

# ----------------- PAGE SETUP & CONFIG -----------------
st.set_page_config(
    page_title="AI Loan Advisory Agent",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Styling headers */
    .main-title {
        background: linear-gradient(135deg, #1D2D50 0%, #133B5C 50%, #1E5F74 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.1rem;
        text-align: left;
    }
    
    .subtitle {
        color: #555555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metrics Card styling */
    .metric-card {
        background: #fdfdfd;
        border: 1px solid #eef2f3;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
        border-color: #1E5F74;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1D2D50;
        margin-bottom: 5px;
    }
    
    .metric-lbl {
        font-size: 0.9rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Expander override */
    .streamlit-expanderHeader {
        background-color: #f7f9fa;
        border-radius: 6px;
    }
    
    /* Citation badges */
    .citation-badge {
        display: inline-block;
        background-color: #e3faf2;
        color: #0ca678;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
        border: 1px solid #c3fae8;
    }
    
    .warning-badge {
        display: inline-block;
        background-color: #fff9db;
        color: #f59f00;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
        border: 1px solid #ffe3e3;
    }

    .fail-badge {
        display: inline-block;
        background-color: #fff5f5;
        color: #fa5252;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
        border: 1px solid #ffe3e3;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- CACHED MODELS -----------------
@st.cache_resource
def load_embeddings_model():
    """Lazy load the sentence transformers model once."""
    return LocalEmbeddingModel()

# ----------------- SESSION STATE INIT -----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = LocalVectorStore()

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Loan Advisory Agent. Ask me questions about loan policies, interest rates, eligibility criteria, and required documents. I can reference our pre-loaded loan schemes or any documents you upload."}
    ]

# ----------------- HELPER FUNCTIONS -----------------
def index_documents(docs: List[Dict[str, Any]], progress_bar=None):
    """
    Chunks, embeds, and indexes list of documents into LocalVectorStore.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    embedding_model = load_embeddings_model()
    
    chunks = []
    texts_to_embed = []
    
    total = len(docs)
    for idx, doc in enumerate(docs):
        split_chunks = splitter.split_text(doc["text"])
        for chunk_idx, chunk_text in enumerate(split_chunks):
            chunk_doc = {
                "id": f"{doc.get('id', 'doc')}_chunk_{chunk_idx}",
                "category": doc.get("category", "General"),
                "title": f"{doc.get('title', 'Document')} (Part {chunk_idx + 1})",
                "text": chunk_text
            }
            chunks.append(chunk_doc)
            texts_to_embed.append(chunk_text)
            
        if progress_bar:
            progress_bar.progress((idx + 1) / total)
            
    if texts_to_embed:
        embeddings = embedding_model.embed_documents(texts_to_embed)
        st.session_state.vector_store.add_documents(chunks, embeddings)
        st.session_state.indexed_docs.extend(chunks)

def initialize_default_policies():
    """Initializes the database with pre-packaged synthetic policies."""
    st.session_state.vector_store.clear()
    st.session_state.indexed_docs = []
    index_documents(LOAN_POLICIES)

# Seed if empty
if not st.session_state.indexed_docs:
    with st.spinner("Initializing default loan policies and embedding model..."):
        initialize_default_policies()


# ----------------- SIDEBAR CONFIG -----------------
st.sidebar.markdown("## 🪙 AI Loan Advisor Settings")

# Model and API keys section
with st.sidebar.expander("🔑 API Key Configurations", expanded=False):
    gemini_key = st.text_input("Gemini API Key", type="password", help="Input your Google Gemini API Key. If empty, local offline extractor will be used.")
    cohere_key = st.text_input("Cohere API Key", type="password", help="Input your Cohere API Key.")

# LLM model settings
st.sidebar.subheader("🤖 Generation Model")
model_option = st.sidebar.selectbox(
    "Choose LLM Agent",
    options=["Offline Grounded Extractor", "Google Gemini", "Cohere"],
    index=0
)

# Map human-readable model to library type
if model_option == "Google Gemini":
    llm_type = "gemini"
    api_key_to_use = gemini_key
    model_name = "gemini-1.5-flash"
elif model_option == "Cohere":
    llm_type = "cohere"
    api_key_to_use = cohere_key
    model_name = "command-r"
else:
    llm_type = "mock"
    api_key_to_use = None
    model_name = None

# Retrieval settings
st.sidebar.subheader("🔍 Retrieval parameters")
top_k_chunks = st.sidebar.slider("Chunks to retrieve (Top-K)", min_value=1, max_value=10, value=4)
alpha_weight = st.sidebar.slider("Hybrid Search Weight (Alpha)", min_value=0.0, max_value=1.0, value=0.6,
                                   help="1.0: Dense Search (Vector match) | 0.0: Sparse Search (Keyword match) | 0.5: Equal combination")
use_reranking = st.sidebar.checkbox("Enable Cross-Encoder Reranking", value=False,
                                    help="Uses a neural cross-encoder model to re-score and re-order the retrieved chunks for higher accuracy.")

# Document upload section
st.sidebar.subheader("📁 Upload Loan Guidelines")
uploaded_files = st.sidebar.file_uploader(
    "Upload Policy PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload and index custom bank policy or advisory guideline PDFs dynamically."
)

if uploaded_files:
    if st.sidebar.button("⚙️ Process and Index PDFs"):
        with st.spinner("Extracting text and building vector representations..."):
            extracted_docs = []
            progress_bar = st.sidebar.progress(0.0)
            for idx, uploaded_file in enumerate(uploaded_files):
                pdf_text = extract_text_from_pdf(uploaded_file)
                if not pdf_text.startswith("Error"):
                    doc_obj = {
                        "id": f"uploaded_pdf_{idx}_{uploaded_file.name.replace(' ', '_')}",
                        "category": "Uploaded Document",
                        "title": uploaded_file.name,
                        "text": pdf_text
                    }
                    extracted_docs.append(doc_obj)
            
            if extracted_docs:
                index_documents(extracted_docs)
                st.sidebar.success(f"Successfully processed {len(extracted_docs)} PDF(s) into database!")
            else:
                st.sidebar.error("Could not parse text from uploaded PDFs.")

# Reset database button
st.sidebar.subheader("♻️ Database Operations")
st.sidebar.info(f"Currently indexed: **{len(st.session_state.indexed_docs)} chunks**.")
if st.sidebar.button("Reset to Default Policies"):
    with st.spinner("Re-initializing default dataset..."):
        initialize_default_policies()
        st.sidebar.success("Database restored to default synthetic policies.")


# ----------------- MAIN INTERFACE -----------------
st.markdown('<h1 class="main-title">AI Loan Advisory Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Natural language retrieval-grounded assistant for bank loan eligibility, EMIs, and policy documents.</p>', unsafe_allow_html=True)

# Create tabs
chat_tab, emi_tab, eligibility_tab = st.tabs(["💬 Advisory Chat", "📊 EMI Calculator", "📋 Eligibility Check"])

# ================= TAB 1: ADVISORY CHAT =================
with chat_tab:
    # Set up retriever
    retriever = HybridRetriever(
        vector_store=st.session_state.vector_store,
        embedding_model=load_embeddings_model(),
        documents=st.session_state.indexed_docs
    )
    
    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "diagnostics" in msg:
                diag = msg["diagnostics"]
                # Display custom citation widgets
                with st.expander("🔍 Citations & Grounding Diagnostics"):
                    # Badges for grounding status
                    g_status = diag["grounding_status"]
                    score = diag["grounding_score"]
                    if g_status == "Pass":
                        st.markdown(f'<span class="citation-badge">Pass</span> **Grounding Score: {score:.2f}** - Answer completely matches reference data.', unsafe_allow_html=True)
                    elif g_status == "Warning":
                        st.markdown(f'<span class="warning-badge">Warning</span> **Grounding Score: {score:.2f}** - Some terms in the answer cannot be fully validated.', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="fail-badge">Fail</span> **Grounding Score: {score:.2f}** - Answer contains heavy hallucinations or unverified figures.', unsafe_allow_html=True)
                    
                    if diag.get("unverified"):
                        st.write(f"⚠️ **Unverified key tokens:** `{', '.join(diag['unverified'])}`")
                        
                    # Show citations
                    st.write("#### 📑 Grounded Citations")
                    if diag.get("citations"):
                        for cit in diag["citations"]:
                            st.write(f"- *\"{cit['sentence_text']}\"* → Cited: **{cit['cited_source_title']}** (Relevance: {cit['overlap_score']:.2f})")
                    else:
                        st.write("No exact line-level matches found. Summary generated from retrieved contexts.")
                        
                    # Show full retrieved text chunks
                    st.write("#### 📄 Retrieved Policy Snippets")
                    for i, chunk in enumerate(diag["retrieved_chunks"]):
                        st.markdown(f"**Source {i+1}: {chunk['title']}** (Score: {chunk.get('score', chunk.get('re_rank_score', 0.0)):.4f})")
                        st.info(chunk["text"])

    # User input
    if query := st.chat_input("Ask a question (e.g. 'What is the minimum credit score for Apex Home Loans?'):"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant policy documents..."):
                # 1. Retrieve candidate chunks
                retrieved_chunks = retriever.hybrid_search(query, top_k=top_k_chunks, alpha=alpha_weight)
                
                # Apply cross encoder re-ranking if enabled
                if use_reranking and retrieved_chunks:
                    try:
                        retrieved_chunks = retriever.re_rank(query, retrieved_chunks, top_k=top_k_chunks)
                    except Exception as re_err:
                        st.warning(f"Reranking model error: {re_err}. Falling back to default retrieval sorting.")
                
            if not retrieved_chunks:
                answer = "I could not find any matching policy documents in the database. Please verify if your document base has been loaded."
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                # 2. Build prompting context
                prompt = build_prompt(query, retrieved_chunks)
                
                with st.spinner("Generating grounded answer..."):
                    # 3. Model Generation
                    try:
                        client = get_llm_client(llm_type, api_key_to_use, model_name)
                        answer = client.generate(prompt)
                    except Exception as gen_err:
                        answer = f"Error during response generation: {str(gen_err)}. Please ensure API key settings are configured if using online LLMs."
                        st.error(answer)
                
                # 4. Response Grounding Validation
                validation_results = validate_response_grounding(answer, retrieved_chunks)
                citations_results = find_sentences_citations(answer, retrieved_chunks)
                
                # Present response
                st.write(answer)
                
                diagnostics_data = {
                    "grounding_status": validation_results["status"],
                    "grounding_score": validation_results["grounding_score"],
                    "unverified": validation_results["unverified_tokens"],
                    "citations": citations_results,
                    "retrieved_chunks": retrieved_chunks
                }
                
                # Display diagnostics right away
                with st.expander("🔍 Citations & Grounding Diagnostics", expanded=True):
                    g_status = validation_results["status"]
                    score = validation_results["grounding_score"]
                    if g_status == "Pass":
                        st.markdown(f'<span class="citation-badge">Pass</span> **Grounding Score: {score:.2f}** - Answer completely matches reference data.', unsafe_allow_html=True)
                    elif g_status == "Warning":
                        st.markdown(f'<span class="warning-badge">Warning</span> **Grounding Score: {score:.2f}** - Some terms in the answer cannot be fully validated.', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="fail-badge">Fail</span> **Grounding Score: {score:.2f}** - Answer contains heavy hallucinations or unverified figures.', unsafe_allow_html=True)
                    
                    if validation_results["unverified_tokens"]:
                        st.write(f"⚠️ **Unverified key tokens:** `{', '.join(validation_results['unverified_tokens'])}`")
                        
                    st.write("#### 📑 Grounded Citations")
                    if citations_results:
                        for cit in citations_results:
                            st.write(f"- *\"{cit['sentence_text']}\"* → Cited: **{cit['cited_source_title']}** (Relevance: {cit['overlap_score']:.2f})")
                    else:
                        st.write("No exact line-level matches found. Summary generated from retrieved contexts.")
                        
                    st.write("#### 📄 Retrieved Policy Snippets")
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**Source {i+1}: {chunk['title']}** (Score: {chunk.get('score', chunk.get('re_rank_score', 0.0)):.4f})")
                        st.info(chunk["text"])
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "diagnostics": diagnostics_data
                })


# ================= TAB 2: EMI CALCULATOR =================
with emi_tab:
    st.markdown("### 📊 Interactive Loan Amortization & EMI Calculator")
    st.write("Quickly visualize monthly commitments, total interest load, and see how the principal balance reduces over the loan timeline.")
    
    col_input, col_metrics = st.columns([1, 2])
    
    with col_input:
        principal_amt = st.slider("Loan Amount (INR)", min_value=50000, max_value=20000000, value=3000000, step=50000, format="INR %d")
        interest_rate = st.slider("Annual Interest Rate (%)", min_value=5.0, max_value=25.0, value=8.75, step=0.05)
        tenure_yrs = st.slider("Tenure (Years)", min_value=1, max_value=30, value=15, step=1)
        
        tenure_months = tenure_yrs * 12
        emi_results = calculate_emi(principal_amt, interest_rate, tenure_months)
        
    with col_metrics:
        # Display key summary cards
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">₹{emi_results['emi']:,}</div>
                <div class="metric-lbl">Monthly EMI</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">₹{emi_results['total_interest']:,}</div>
                <div class="metric-lbl">Total Interest</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">₹{emi_results['total_payment']:,}</div>
                <div class="metric-lbl">Total Payment</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Draw Area Chart representing Balance reduction
        st.write("")
        st.write("#### 📈 Outstanding Loan Balance Over Time")
        schedule_df = pd.DataFrame(emi_results["schedule"])
        if not schedule_df.empty:
            # We map Month to Year
            schedule_df["Year"] = (schedule_df["Month"] / 12).round(1)
            balance_chart_data = schedule_df.set_index("Year")[["Balance"]]
            st.area_chart(balance_chart_data, color="#1e5f74")
            
    # Show schedule table
    with st.expander("📋 View Complete Amortization Schedule Table"):
        if not schedule_df.empty:
            st.dataframe(
                schedule_df[["Month", "EMI", "Principal", "Interest", "Balance"]],
                use_container_width=True,
                height=300
            )
            # Enable CSV download
            csv_buffer = io.StringIO()
            schedule_df[["Month", "EMI", "Principal", "Interest", "Balance"]].to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Amortization Schedule CSV",
                data=csv_buffer.getvalue(),
                file_name=f"amortization_schedule_{principal_amt}.csv",
                mime="text/csv"
            )


# ================= TAB 3: ELIGIBILITY CHECK =================
with eligibility_tab:
    st.markdown("### 📋 Credit Eligibility & Risk Assessment")
    st.write("Assess applicant profile status using standard rule-based FOIR calculations or our trained Machine Learning classification model.")
    
    check_mode = st.radio(
        "Select Evaluation Mode:",
        options=["Rule-Based FOIR Guidelines", "Machine Learning Underwriting Predictor (Data Science Model)"],
        horizontal=True
    )
    
    if check_mode == "Rule-Based FOIR Guidelines":
        st.info("ℹ️ **Rule-Based Mode:** Evaluates eligibility using the Fixed Income to Obligations Ratio (FOIR) standard. It calculates if combined debt commitments exceed 50-60% of monthly income.")
        e_col1, e_col2 = st.columns([1, 1])
        
        with e_col1:
            net_monthly_salary = st.number_input("Net Monthly Income (INR)", min_value=10000, value=65000, step=1000, key="rb_salary")
            existing_monthly_emi = st.number_input("Existing Monthly EMI Commitments (INR)", min_value=0, value=8000, step=500, key="rb_emi")
            target_loan_amt = st.number_input("Desired Loan Amount (INR)", min_value=10000, value=2500000, step=50000, key="rb_loan")
            est_rate = st.slider("Assumed Interest Rate (% p.a.)", min_value=5.0, max_value=25.0, value=8.75, step=0.1, key="rb_rate")
            est_tenure = st.slider("Assumed Tenure (Years)", min_value=1, max_value=30, value=20, step=1, key="rb_tenure")
            
            # Trigger check
            eligibility = check_eligibility(
                monthly_income=net_monthly_salary,
                existing_emi=existing_monthly_emi,
                desired_loan=target_loan_amt,
                rate_pa=est_rate,
                tenure_months=est_tenure * 12
            )
            
        with e_col2:
            st.write("#### 🛡️ Assessment Results")
            
            status = eligibility["status"]
            if eligibility["eligible"]:
                st.success(f"🎉 **Status: {status}** - You are eligible for this loan scheme!")
            else:
                st.error(f"❌ **Status: {status}** - The desired loan exceeds safety thresholds.")
                
            st.info(f"💡 **Assessment Details:** {eligibility['reason']}")
            
            st.write("#### 📐 Mathematical Breakdown")
            breakdown_data = {
                "Factor": [
                    "Net Monthly Income",
                    "Allowed Debt Obligation Ratio (FOIR Limit)",
                    "Maximum Permissible Combined EMI",
                    "Existing Monthly EMIs",
                    "Maximum Available EMI Budget",
                    "Desired Loan Required EMI",
                    "Estimated Max Eligible Loan Amount"
                ],
                "Value": [
                    f"INR {net_monthly_salary:,.2f}",
                    f"{eligibility['foir_percentage']}%",
                    f"INR {net_monthly_salary * (eligibility['foir_percentage']/100):,.2f}",
                    f"INR {existing_monthly_emi:,.2f}",
                    f"INR {eligibility['max_emi_allowed']:,.2f}",
                    f"INR {eligibility['desired_emi']:,.2f}",
                    f"INR {eligibility['max_loan_eligible']:,.2f}"
                ]
            }
            st.table(pd.DataFrame(breakdown_data))
            
    else:
        # Machine Learning Mode
        model_package = load_ml_model()
        if model_package is None:
            st.warning("⚠️ **Predictive Model Artifact Not Found:** Please run the training script (`python src/train_ml.py`) to generate the classifier model pickle.")
        else:
            st.info("🧠 **Data Science Predictive Underwriting Model:** A Random Forest Classifier trained on 4,269 historical cases. Predicts loan approval probability based on Credit (CIBIL) score, asset valuations, and income ratio.")
            
            e_col1, e_col2 = st.columns([1, 1])
            
            with e_col1:
                income_annum = st.number_input("Annual Income (INR)", min_value=10000, value=780000, step=10000, key="ml_income")
                loan_amount = st.number_input("Desired Loan Amount (INR)", min_value=10000, value=2500000, step=50000, key="ml_loan")
                loan_term = st.slider("Loan Tenure (Years)", min_value=1, max_value=20, value=10, step=1, key="ml_term")
                cibil_score = st.slider("Credit (CIBIL) Score", min_value=300, max_value=900, value=720, step=5)
                no_of_dependents = st.slider("Number of Dependents", min_value=0, max_value=10, value=2)
                
                c1, c2 = st.columns(2)
                with c1:
                    education = st.selectbox("Education Status", ["Graduate", "Not Graduate"])
                with c2:
                    self_employed = st.selectbox("Self Employment Status", ["No", "Yes"])
                
                with st.expander("💼 Financial Asset Valuations (INR)", expanded=False):
                    res_assets = st.number_input("Residential Assets Value", min_value=0, value=1500000, step=50000)
                    com_assets = st.number_input("Commercial Assets Value", min_value=0, value=500000, step=50000)
                    lux_assets = st.number_input("Luxury Assets Value (e.g. cars, jewelry)", min_value=0, value=1000000, step=50000)
                    bank_assets = st.number_input("Bank Deposits / Liquid Assets Value", min_value=0, value=300000, step=10000)
                    
            with e_col2:
                features_list = model_package["features"]
                model = model_package["model"]
                
                input_df = pd.DataFrame([{
                    'no_of_dependents': no_of_dependents,
                    'education': 1 if education == "Graduate" else 0,
                    'self_employed': 1 if self_employed == "Yes" else 0,
                    'income_annum': income_annum,
                    'loan_amount': loan_amount,
                    'loan_term': loan_term,
                    'cibil_score': cibil_score,
                    'residential_assets_value': res_assets,
                    'commercial_assets_value': com_assets,
                    'luxury_assets_value': lux_assets,
                    'bank_asset_value': bank_assets
                }])
                
                # Make sure columns are matched in order
                input_df = input_df[features_list]
                
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                prob_approved = probabilities[1]
                
                st.write("#### 🛡️ Machine Learning Underwriting Output")
                
                if prediction == 1:
                    st.success(f"🎉 **Prediction: APPROVED** (Confidence: {prob_approved * 100:.1f}%)")
                    risk_label = "Low Risk" if prob_approved >= 0.80 else "Medium Risk"
                    risk_color = "green" if prob_approved >= 0.80 else "orange"
                else:
                    st.error(f"❌ **Prediction: REJECTED** (Confidence: {(1 - prob_approved) * 100:.1f}%)")
                    risk_label = "High Risk"
                    risk_color = "red"
                    
                st.markdown(f"**Risk Categorization:** <span style='color:{risk_color}; font-weight:bold;'>{risk_label}</span>", unsafe_allow_html=True)
                
                st.write("")
                st.write("**Approval Probability Score**")
                st.progress(prob_approved)
                
                st.write("")
                st.write("#### 🔍 Model Key Decision Drivers")
                
                feat_importances = model_package["feature_importances"]
                name_mapping = {
                    'cibil_score': 'CIBIL Credit Score',
                    'loan_amount': 'Desired Loan Amount',
                    'income_annum': 'Annual Income',
                    'luxury_assets_value': 'Luxury Assets Val',
                    'residential_assets_value': 'Residential Assets Val',
                    'bank_asset_value': 'Bank Assets Val',
                    'commercial_assets_value': 'Commercial Assets Val',
                    'loan_term': 'Loan Term (Years)',
                    'no_of_dependents': 'No. of Dependents',
                    'education': 'Education Level',
                    'self_employed': 'Self Employment Status'
                }
                display_importances = {}
                for k, v in feat_importances.items():
                    disp_name = name_mapping.get(k, k)
                    display_importances[disp_name] = v
                
                sorted_imp = sorted(display_importances.items(), key=lambda x: x[1], reverse=True)
                imp_df = pd.DataFrame(sorted_imp, columns=["Key Decision Driver", "Relative Decision Weight"])
                st.bar_chart(imp_df.head(6).set_index("Key Decision Driver"), color="#1e5f74")
                
                st.markdown(f"""
                <hr style='margin: 15px 0 10px 0;'>
                <div style='font-size: 0.75rem; color: gray;'>
                    <b>ML Model Validation Diagnostics:</b><br>
                    Model Type: Random Forest Classifier (100 Estimators, Depth 10)<br>
                    Training Set Accuracy: {model_package['metrics']['accuracy'] * 100:.2f}% | Area Under ROC Curve: {model_package['metrics']['roc_auc']:.4f}
                </div>
                """, unsafe_allow_html=True)
