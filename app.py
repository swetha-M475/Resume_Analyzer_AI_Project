"""
Resume Analyzer — Main Streamlit Application
AI-powered Resume Analysis with JD Matching & Chatbot

Features:
    - Upload PDF resumes and extract text
    - Match resume against Job Descriptions with scoring
    - Interactive chatbot for resume Q&A
"""

import os
# Force pure-python implementation of Protobuf to bypass Streamlit Cloud C-extension conflicts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import time
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from src.resume_parser import extract_text_from_pdf, clean_resume_text, chunk_text
from src.embeddings import get_embeddings, create_vector_store, get_retriever
from src.jd_matcher import match_resume_with_jd, extract_match_score
from src.chatbot import get_chat_response
from prompts.chatbot_prompt import CHATBOT_GREETING

# ─── Load Environment Variables ───────────────────────────────────────────────
load_dotenv()

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Analyzer AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global Theme ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header Styling ── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }

    /* ── Card Styling ── */
    .info-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    .info-card h3 {
        color: #e0e0ff;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 0.8rem 0;
    }

    .info-card p {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.6;
    }

    /* ── Score Display ── */
    .score-container {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }

    .score-value {
        font-size: 4rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }

    .score-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .score-high { color: #4ade80; }
    .score-medium { color: #facc15; }
    .score-low { color: #f87171; }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .badge-success {
        background: rgba(74, 222, 128, 0.15);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }

    .badge-warning {
        background: rgba(250, 204, 21, 0.15);
        color: #facc15;
        border: 1px solid rgba(250, 204, 21, 0.3);
    }

    .badge-info {
        background: rgba(96, 165, 250, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ─────────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "resumes": {},              # {file_name: {"text": cleaned_text, "chunks": chunks}}
        "vector_store": None,
        "retriever": None,
        "embeddings": None,
        "llm": None,
        "chat_history": [],
        "resume_uploaded": False,
        "file_names": [],           # List of uploaded file names
        "jd_analysis_results": {},  # {file_name: {"score": score, "analysis": analysis}}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─── LLM Initialization ──────────────────────────────────────────────────────
@st.cache_resource
def initialize_llm():
    """Initialize the HuggingFace LLM (cached across reruns)."""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token or hf_token == "hf_your_token_here":
        return None

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=hf_token,
        task="conversational",
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    return llm


@st.cache_resource
def initialize_embeddings():
    """Initialize HuggingFace embeddings (cached across reruns)."""
    return get_embeddings()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with app info and status."""
    with st.sidebar:
        st.markdown("## 📄 Resume Analyzer AI")
        st.markdown("---")

        # Status indicators
        st.markdown("### ⚙️ System Status")

        # LLM Status
        if st.session_state.llm:
            st.markdown('<span class="status-badge badge-success">✅ LLM Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-warning">⚠️ LLM Not Connected</span>', unsafe_allow_html=True)
            st.caption("Add your HuggingFace token to `.env`")

        st.markdown("")

        # Resume Status
        num_resumes = len(st.session_state.file_names)
        if num_resumes > 0:
            st.markdown(f'<span class="status-badge badge-success">✅ {num_resumes} Resume(s) Loaded</span>', unsafe_allow_html=True)
            for name in st.session_state.file_names:
                st.caption(f"📎 {name}")
        else:
            st.markdown('<span class="status-badge badge-info">📤 Upload Resumes</span>', unsafe_allow_html=True)

        st.markdown("")

        # Vector Store Status
        if st.session_state.vector_store:
            st.markdown('<span class="status-badge badge-success">✅ Vector Store Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-info">🔲 No Vectors Yet</span>', unsafe_allow_html=True)

        st.markdown("---")

        # About section
        st.markdown("### 📌 About")
        st.markdown("""
        This app uses:
        - **LangChain** for orchestration
        - **ChromaDB** for vector storage
        - **HuggingFace** for embeddings & LLM
        - **Streamlit** for the UI
        """)

        st.markdown("---")

        # Clear data button
        if st.button("🗑️ Clear All Data", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()


# ─── Tab 1: Resume Upload ────────────────────────────────────────────────────
def render_upload_tab():
    """Render the Resume Upload tab."""
    st.markdown("### 📤 Upload Your Resumes")
    st.markdown("Upload one or multiple PDF resumes to get started. The text will be extracted, chunked, and stored in a shared vector database.")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose PDF resumes",
        type=["pdf"],
        key="resume_uploader",
        accept_multiple_files=True,
        help="Upload one or multiple resumes in PDF format",
    )

    if uploaded_files:
        new_files_processed = False
        new_names = [f.name for f in uploaded_files]

        # Detect if any file was removed from the uploader
        files_to_remove = [name for name in list(st.session_state.resumes.keys()) if name not in new_names]
        if files_to_remove:
            for name in files_to_remove:
                del st.session_state.resumes[name]
                if name in st.session_state.jd_analysis_results:
                    del st.session_state.jd_analysis_results[name]
            st.session_state.file_names = [n for n in st.session_state.file_names if n not in files_to_remove]
            new_files_processed = True

        # Process any new files
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.resumes:
                with st.spinner(f"🔄 Processing {uploaded_file.name}..."):
                    try:
                        # Extract & clean text
                        raw_text = extract_text_from_pdf(uploaded_file)
                        cleaned_text = clean_resume_text(raw_text)

                        # Chunk text (include source filename as metadata)
                        chunks = chunk_text(cleaned_text, source=uploaded_file.name)

                        # Save to session state
                        st.session_state.resumes[uploaded_file.name] = {
                            "text": cleaned_text,
                            "chunks": chunks
                        }
                        if uploaded_file.name not in st.session_state.file_names:
                            st.session_state.file_names.append(uploaded_file.name)
                        new_files_processed = True

                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

        if new_files_processed:
            # Rebuild Vector Store from all current resumes
            if st.session_state.resumes:
                with st.spinner("🧠 Updating vector database..."):
                    try:
                        if st.session_state.embeddings is None:
                            st.session_state.embeddings = initialize_embeddings()

                        all_chunks = []
                        for name, data in st.session_state.resumes.items():
                            all_chunks.extend(data["chunks"])

                        vector_store = create_vector_store(all_chunks, st.session_state.embeddings)
                        retriever = get_retriever(vector_store)

                        st.session_state.vector_store = vector_store
                        st.session_state.retriever = retriever
                        st.session_state.resume_uploaded = True
                        st.session_state.chat_history = []  # Reset chat on update
                        st.success("✅ Vector database updated successfully with all resumes!")
                    except Exception as e:
                        st.error(f"❌ Error updating vector database: {str(e)}")
            else:
                st.session_state.vector_store = None
                st.session_state.retriever = None
                st.session_state.resume_uploaded = False

    # Show list of uploaded resumes and text
    if st.session_state.resumes:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📋 View Extracted Resume")
            selected_resume = st.selectbox(
                "Select a resume to view",
                options=st.session_state.file_names,
                key="view_resume_selector"
            )

            if selected_resume:
                with st.expander(f"View full text of {selected_resume}", expanded=True):
                    st.text(st.session_state.resumes[selected_resume]["text"])

        with col2:
            st.markdown("### 📊 Database Statistics")
            total_chars = sum(len(data["text"]) for data in st.session_state.resumes.values())
            total_words = sum(len(data["text"].split()) for data in st.session_state.resumes.values())
            total_chunks = sum(len(data["chunks"]) for data in st.session_state.resumes.values())

            st.metric("Total Resumes", len(st.session_state.resumes))
            st.metric("Total Characters", f"{total_chars:,}")
            st.metric("Total Word Count", f"{total_words:,}")
            st.metric("Total Chunks", f"{total_chunks}")


# ─── Tab 2: JD Matching ──────────────────────────────────────────────────────
def render_matching_tab():
    """Render the JD Matching tab."""
    st.markdown("### 🎯 Job Description Matching")
    st.markdown("Paste a job description below to analyze and compare all uploaded resumes.")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if not st.session_state.resume_uploaded:
        st.warning("⚠️ Please upload at least one resume in the **📄 Resume Upload** tab.")
        return

    if not st.session_state.llm:
        st.error("❌ LLM not connected. Please add your HuggingFace token to the `.env` file and restart.")
        return

    # JD Input
    jd_text = st.text_area(
        "Paste the Job Description here",
        height=200,
        placeholder="Paste the full job description here...\n\nExample:\nWe are looking for a Python Developer with 3+ years of experience in Django, REST APIs, and PostgreSQL...",
        key="jd_input",
    )

    if st.button("🔍 Analyze & Compare All Resumes", use_container_width=True, type="primary"):
        if not jd_text.strip():
            st.warning("⚠️ Please paste a job description first.")
            return

        # Initialize dictionary to store match analysis for each resume
        st.session_state.jd_analysis_results = {}

        # Progress bar for multiple resumes
        num_resumes = len(st.session_state.resumes)
        progress_bar = st.progress(0, text="Starting analysis...")

        for idx, (name, data) in enumerate(st.session_state.resumes.items()):
            progress_bar.progress(idx / num_resumes, text=f"🧠 Analyzing {name} ({idx+1}/{num_resumes})...")
            try:
                analysis = match_resume_with_jd(
                    data["text"],
                    jd_text,
                    st.session_state.llm,
                )
                score = extract_match_score(analysis)
                st.session_state.jd_analysis_results[name] = {
                    "score": score,
                    "analysis": analysis
                }
            except Exception as e:
                st.session_state.jd_analysis_results[name] = {
                    "score": -1,
                    "analysis": f"❌ Error during analysis: {str(e)}"
                }

        progress_bar.progress(100, text="✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()

    # Display results if they exist in session state
    if st.session_state.jd_analysis_results:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🏆 Candidate Leaderboard")

        # Build leaderboard list sorted by score descending
        leaderboard = []
        for name, res in st.session_state.jd_analysis_results.items():
            leaderboard.append({
                "Candidate / File": name,
                "Score": res["score"] if res["score"] >= 0 else "N/A"
            })

        # Sort by score descending (handling N/A/errors at the bottom)
        leaderboard.sort(key=lambda x: x["Score"] if isinstance(x["Score"], int) else -1, reverse=True)

        # Show leaderboard table
        st.table(leaderboard)

        # Choose a resume to view detailed analysis
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔍 Detailed Feedback")
        selected_detailed = st.selectbox(
            "Select a candidate to view their detailed feedback",
            options=st.session_state.file_names
        )

        if selected_detailed and selected_detailed in st.session_state.jd_analysis_results:
            res = st.session_state.jd_analysis_results[selected_detailed]
            score = res["score"]
            analysis = res["analysis"]

            if score >= 0:
                col1, col2 = st.columns([1, 2])
                with col1:
                    if score >= 70:
                        score_class = "score-high"
                        score_emoji = "🟢"
                    elif score >= 40:
                        score_class = "score-medium"
                        score_emoji = "🟡"
                    else:
                        score_class = "score-low"
                        score_emoji = "🔴"

                    st.markdown(f"""
                    <div class="score-container">
                        <p class="score-value {score_class}">{score}</p>
                        <p class="score-label">{score_emoji} Match Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("")
                    st.progress(score / 100)

                with col2:
                    st.markdown(analysis)
            else:
                st.markdown(analysis)


# ─── Tab 3: Chatbot ──────────────────────────────────────────────────────────
def render_chatbot_tab():
    """Render the Chatbot tab."""
    st.markdown("### 💬 Resume Chatbot")
    st.markdown("Ask me anything about your resume!")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if not st.session_state.resume_uploaded:
        st.warning("⚠️ Please upload a resume first in the **📄 Resume Upload** tab.")
        return

    if not st.session_state.llm:
        st.error("❌ LLM not connected. Please add your HuggingFace token to the `.env` file and restart.")
        return

    # Display chat history
    if not st.session_state.chat_history:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(CHATBOT_GREETING)

    for human_msg, ai_msg in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.markdown(human_msg)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(ai_msg)

    # Chat input
    if prompt := st.chat_input("Ask about your resume...", key="chat_input"):
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    response = get_chat_response(
                        retriever=st.session_state.retriever,
                        llm=st.session_state.llm,
                        question=prompt,
                        chat_history=st.session_state.chat_history,
                    )
                    st.markdown(response)

                    # Save to history
                    st.session_state.chat_history.append((prompt, response))

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)


# ─── Main Application ────────────────────────────────────────────────────────
def main():
    """Main application entry point."""
    # Initialize LLM
    if st.session_state.llm is None:
        st.session_state.llm = initialize_llm()

    # Render sidebar
    render_sidebar()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Resume Analyzer AI</h1>
        <p>Upload your resume, match it with job descriptions, and chat with AI about your profile</p>
    </div>
    """, unsafe_allow_html=True)

    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "📄 Resume Upload",
        "🎯 JD Matching",
        "💬 Chatbot",
    ])

    with tab1:
        render_upload_tab()

    with tab2:
        render_matching_tab()

    with tab3:
        render_chatbot_tab()


if __name__ == "__main__":
    main()
