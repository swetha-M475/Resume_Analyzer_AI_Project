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

from src.resume_parser import extract_text_from_file, clean_resume_text, chunk_text
from src.embeddings import get_embeddings, create_vector_store, get_retriever
from src.jd_matcher import match_resume_with_jd, extract_match_score
from src.chatbot import get_chat_response
from src.career_path_optimizer import (
    extract_skills_from_text, compare_algorithms, SKILL_DATABASE,
    hill_climbing_optimizer, a_star_optimizer, calculate_match_score as calc_skill_match
)
from src.job_route_optimizer import (
    solve_tsp, get_city_coordinates, CITY_COORDINATES
)
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

    /* ── Algorithm Card ── */
    .algo-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin: 0.8rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    .algo-card h4 {
        color: #a78bfa;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
    }

    .algo-card p {
        color: rgba(255, 255, 255, 0.65);
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Path Step ── */
    .path-step {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.6rem 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 3px solid #6366f1;
    }

    .path-step.prereq {
        border-left-color: #f59e0b;
    }

    /* ── Route Card ── */
    .route-segment {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.8rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        margin: 0.2rem 0;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
    }

    .comparison-box {
        background: linear-gradient(135deg, #0f172a, #1e1b4b);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin: 1rem 0;
    }

    .comparison-box h4 {
        color: #818cf8;
        margin: 0 0 0.8rem 0;
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
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "openrouter_model": os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
        "llm_provider_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─── LLM Initialization ──────────────────────────────────────────────────────
@st.cache_resource
def get_cached_llm(provider: str, api_key: str, model_name: str):
    """Create and cache LLM instance based on provider and credentials."""
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "nvidia/nemotron-3.5-lightning:free",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Resume Analyzer AI",
            },
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )
    elif provider == "huggingface":
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            huggingfacehub_api_token=api_key,
            task="conversational",
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        return ChatHuggingFace(llm=llm_endpoint)
    return None


def initialize_llm():
    """Initialize LLM prioritizing OpenRouter, then Gemini (1.5 Flash), then HuggingFace."""
    # 1. OpenRouter (Primary & Recommended - eliminates rate limit bottlenecks)
    openrouter_key = st.session_state.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY", "")
    openrouter_key = openrouter_key.strip() if openrouter_key else ""
    openrouter_model = st.session_state.get("openrouter_model") or os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
    openrouter_model = openrouter_model.strip() if openrouter_model else "nvidia/nemotron-3.5-lightning:free"

    if openrouter_key and openrouter_key != "sk-or-v1-your_openrouter_api_key_here":
        try:
            llm = get_cached_llm("openrouter", openrouter_key, openrouter_model)
            model_short = openrouter_model.split("/")[-1].replace(":free", " (free)")
            st.session_state["llm_provider_name"] = f"OpenRouter [{model_short}]"
            return llm
        except Exception as e:
            st.warning(f"Could not initialize OpenRouter LLM: {e}")

    # 2. Gemini fallback (using stable 1.5-flash)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key.strip() and gemini_key != "your_gemini_api_key_here":
        try:
            llm = get_cached_llm("gemini", gemini_key.strip(), "gemini-1.5-flash")
            st.session_state["llm_provider_name"] = "Google Gemini (1.5 Flash)"
            return llm
        except Exception as e:
            st.warning(f"Could not initialize Gemini LLM: {e}")

    # 3. HuggingFace fallback
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token and hf_token.strip() and hf_token != "hf_your_token_here":
        try:
            llm = get_cached_llm("huggingface", hf_token.strip(), "Llama-3.1-8B")
            st.session_state["llm_provider_name"] = "HuggingFace (Llama-3.1-8B)"
            return llm
        except Exception as e:
            st.warning(f"Could not initialize HuggingFace LLM: {e}")

    st.session_state["llm_provider_name"] = None
    return None


@st.cache_resource
def initialize_embeddings():
    """Initialize HuggingFace embeddings (cached across reruns)."""
    return get_embeddings()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with app info, model configuration, and status."""
    with st.sidebar:
        st.markdown("## 📄 Resume Analyzer AI")
        st.markdown("---")

        # Status indicators
        st.markdown("### ⚙️ System Status")

        # LLM Status
        provider_name = st.session_state.get("llm_provider_name")
        if st.session_state.llm:
            badge_label = f"✅ LLM: {provider_name}" if provider_name else "✅ LLM Connected"
            st.markdown(f'<span class="status-badge badge-success">{badge_label}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-warning">⚠️ LLM Not Connected</span>', unsafe_allow_html=True)
            st.caption("Provide an OpenRouter API key below or in `.env`")

        # OpenRouter Model Settings in Sidebar
        with st.expander("🔑 AI Model Settings (OpenRouter)", expanded=(st.session_state.llm is None)):
            st.markdown(
                "Use OpenRouter to bypass rate limits. "
                "Get your key at [openrouter.ai/keys](https://openrouter.ai/keys)."
            )
            key_input = st.text_input(
                "OpenRouter API Key",
                type="password",
                value=st.session_state.get("openrouter_api_key", ""),
                placeholder="sk-or-v1-...",
                help="Enter your OpenRouter API key here or define OPENROUTER_API_KEY in .env"
            )

            model_options = [
                "nvidia/nemotron-3.5-lightning:free",
                "google/gemma-4-26b-a4b-it:free",
                "deepseek/deepseek-chat",
                "meta-llama/llama-3.3-70b-instruct",
                "Custom...",
            ]

            curr_model = st.session_state.get("openrouter_model", "nvidia/nemotron-3.5-lightning:free")
            selected_idx = model_options.index(curr_model) if curr_model in model_options else len(model_options) - 1

            chosen_model = st.selectbox(
                "Model",
                options=model_options,
                index=selected_idx,
                help="Select free or paid OpenRouter model"
            )
            if chosen_model == "Custom...":
                custom_model = st.text_input("Custom Model ID", value=curr_model if curr_model not in model_options else "")
                final_model = custom_model.strip() if custom_model.strip() else "meta-llama/llama-3.3-70b-instruct:free"
            else:
                final_model = chosen_model

            if st.button("💾 Apply & Connect", use_container_width=True, type="primary"):
                st.session_state.openrouter_api_key = key_input.strip()
                st.session_state.openrouter_model = final_model.strip()
                st.session_state.llm = initialize_llm()
                st.rerun()

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
        - **OpenRouter / HuggingFace** for LLM
        - **HuggingFace** for embeddings
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
    st.markdown("Upload one or multiple resumes (PDF, DOCX, or TXT) to get started. The text will be extracted, chunked, and stored in a shared vector database.")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        key="resume_uploader",
        accept_multiple_files=True,
        help="Upload one or multiple resumes in PDF, DOCX, or TXT format",
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
                        raw_text = extract_text_from_file(uploaded_file)
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
        st.error("❌ LLM not connected. Please enter your OpenRouter API key in the sidebar settings or `.env` file.")
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
        st.markdown("### 🏆 Resume Leaderboard")

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
        st.error("❌ LLM not connected. Please enter your OpenRouter API key in the sidebar settings or `.env` file.")
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


# ─── Tab 4: Career Path Optimizer (Hill Climbing + A*) ───────────────────────
def render_career_path_tab():
    """Render the Career Path Optimizer tab using Hill Climbing and A* algorithms."""
    st.markdown("### 🧗 Career Path Optimizer")
    st.markdown(
        "Uses **Hill Climbing** (greedy) and **A\\* Search** (optimal) algorithms "
        "to find the best skill learning path from your current resume to a target job role."
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Algorithm Explanation ──
    with st.expander("📖 How It Works — State Space Search", expanded=False):
        st.markdown("""
**State Space Search** models the career path as a search problem:

| Concept | Mapping |
|---------|---------|
| **State** | Set of skills the candidate currently has |
| **Goal State** | Set of skills required for the target job |
| **Operators** | Learning a new skill (if prerequisites are met) |
| **Cost** | Estimated days to learn each skill |
| **Heuristic** | Sum of effort for remaining missing skills |

**Hill Climbing** 🏔️
- Greedy algorithm — always picks the skill with the best *value* (improvement/effort)
- Fast but may get stuck at local maxima
- Cannot backtrack once a choice is made

**A\\* Search** ⭐
- Uses `f(n) = g(n) + h(n)` where g = effort spent, h = estimated remaining effort
- Guarantees the **optimal** (minimum total effort) path
- Explores more nodes but finds the best solution
        """)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Step 1: Get Current Skills ──
    st.markdown("#### Step 1: Your Current Skills")

    current_skills = set()
    auto_detected = set()

    if st.session_state.resume_uploaded and st.session_state.resumes:
        # Auto-detect skills from resume
        for name, data in st.session_state.resumes.items():
            detected = extract_skills_from_text(data["text"])
            auto_detected.update(detected)

        if auto_detected:
            st.success(f"✅ Auto-detected **{len(auto_detected)}** skills from your resume(s)")

            # Show detected skills in a nice layout
            cols = st.columns(4)
            for idx, skill in enumerate(sorted(auto_detected)):
                cat = SKILL_DATABASE.get(skill, {}).get("category", "Other")
                cols[idx % 4].markdown(f"- 🔹 **{skill.title()}** `{cat}`")
    else:
        st.info("💡 Upload a resume in the **📄 Resume Upload** tab to auto-detect skills, "
                "or manually enter them below.")

    # Manual skill entry
    manual_input = st.text_input(
        "Add/modify skills (comma-separated)",
        value=", ".join(sorted(auto_detected)) if auto_detected else "",
        placeholder="python, java, html, css, react, sql, git...",
        help="Edit the auto-detected skills or enter your own",
        key="career_current_skills"
    )

    if manual_input.strip():
        current_skills = {s.strip().lower() for s in manual_input.split(",") if s.strip()}
    else:
        current_skills = auto_detected

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Step 2: Define Target Skills ──
    st.markdown("#### Step 2: Target Job Skills")
    st.markdown("Enter the skills required for your target job role.")

    # Preset role templates
    role_presets = {
        "Custom (Enter manually)": "",
        "Full Stack Developer": "python, javascript, html, css, react, node.js, sql, git, docker, rest api, mongodb",
        "Data Scientist": "python, statistics, machine learning, deep learning, pandas, numpy, scikit-learn, sql, data visualization, tensorflow",
        "ML Engineer": "python, machine learning, deep learning, tensorflow, pytorch, docker, aws, git, mlops, data structures, algorithms",
        "Frontend Developer": "html, css, javascript, typescript, react, next.js, git, rest api",
        "Backend Developer": "python, django, rest api, sql, postgresql, docker, git, linux, redis",
        "DevOps Engineer": "linux, docker, kubernetes, aws, terraform, ci/cd, git, jenkins, ansible, python",
        "Mobile Developer": "java, kotlin, android, flutter, git, rest api, firebase",
        "Cloud Architect": "aws, azure, gcp, docker, kubernetes, terraform, networking, linux, system design, ci/cd",
        "Cybersecurity Analyst": "networking, linux, cybersecurity, ethical hacking, python, sql",
        "AI/NLP Engineer": "python, machine learning, deep learning, nlp, pytorch, langchain, prompt engineering, rag, generative ai",
    }

    selected_preset = st.selectbox(
        "Choose a target role template (or enter custom)",
        options=list(role_presets.keys()),
        key="career_role_preset"
    )

    default_target = role_presets.get(selected_preset, "")

    jd_input_text = st.text_area(
        "Paste Job Description to extract target skills",
        value=default_target,
        placeholder="Paste JD here or manually type comma-separated skills...",
        height=150,
        key="career_target_jd"
    )

    target_skills = set()
    if jd_input_text.strip():
        # First, try to see if it's just a comma-separated list
        if "," in jd_input_text and len(jd_input_text.split("\n")) < 3:
            target_skills = {s.strip().lower() for s in jd_input_text.split(",") if s.strip()}
        else:
            # If it looks like a full JD, extract skills using the NLP matcher
            target_skills = extract_skills_from_text(jd_input_text)
            
            if target_skills:
                st.success(f"✅ Auto-detected **{len(target_skills)}** skills from the Job Description")
                cols = st.columns(4)
                for idx, skill in enumerate(sorted(target_skills)):
                    cat = SKILL_DATABASE.get(skill, {}).get("category", "Other")
                    cols[idx % 4].markdown(f"- 🎯 **{skill.title()}** `{cat}`")
            else:
                st.warning("Could not auto-detect any known skills from the provided text.")

    # Validate skills
    unknown_current = current_skills - set(SKILL_DATABASE.keys())
    unknown_target = target_skills - set(SKILL_DATABASE.keys())

    if unknown_current:
        st.warning(f"⚠️ Unknown skills in your set (will be ignored): {', '.join(unknown_current)}")
        current_skills -= unknown_current

    if unknown_target:
        st.warning(f"⚠️ Unknown skills in target (will be ignored): {', '.join(unknown_target)}")
        target_skills -= unknown_target

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Step 3: Run Algorithms ──
    if st.button("🚀 Find Optimal Learning Path", use_container_width=True, type="primary",
                 key="career_run_btn"):
        if not current_skills:
            st.warning("⚠️ Please enter your current skills.")
            return
        if not target_skills:
            st.warning("⚠️ Please enter target job skills.")
            return

        missing = target_skills - current_skills
        if not missing:
            st.success("🎉 You already have ALL the target skills! No learning path needed.")
            return

        st.markdown(f"**Current Skills:** {len(current_skills)} | "
                    f"**Target Skills:** {len(target_skills)} | "
                    f"**Missing:** {len(missing)}")

        with st.spinner("🧠 Running Hill Climbing and A* algorithms..."):
            results = compare_algorithms(current_skills, target_skills)
            st.session_state["career_results"] = results

    # ── Display Results ──
    if "career_results" in st.session_state and st.session_state["career_results"]:
        results = st.session_state["career_results"]
        hill = results["hill_climbing"]
        astar = results["a_star"]
        comparison = results["comparison"]

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Comparison Summary ──
        st.markdown("### 📊 Algorithm Comparison")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hill Climbing Effort", f"{hill['total_effort_days']} days",
                      delta=f"{hill['total_skills_to_learn']} skills")
        with col2:
            st.metric("A* Optimal Effort", f"{astar['total_effort_days']} days",
                      delta=f"{astar['total_skills_to_learn']} skills")
        with col3:
            saved = comparison['effort_saved_by_astar']
            st.metric("A* Saves", f"{saved} days",
                      delta=f"{comparison['astar_efficiency_gain_percent']}% more efficient",
                      delta_color="normal")

        st.markdown(f"""
        <div class="comparison-box">
            <h4>🔍 Key Insight</h4>
            <p style="color: rgba(255,255,255,0.75); margin: 0;">
                {comparison['summary']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Side-by-side paths ──
        st.markdown("### 🛤️ Learning Paths")

        tab_hc, tab_astar = st.tabs(["🏔️ Hill Climbing Path", "⭐ A* Optimal Path"])

        for tab, result, label in [(tab_hc, hill, "Hill Climbing"), (tab_astar, astar, "A*")]:
            with tab:
                if not result["path"]:
                    st.warning(f"⚠️ {label} could not find a complete path.")
                    continue

                st.markdown(f"**Score: {result['initial_score']}% → {result['final_score']}%** | "
                            f"**Total Effort: {result['total_effort_days']} days** | "
                            f"**Skills to Learn: {result['total_skills_to_learn']}**")

                st.progress(result["final_score"] / 100)

                # Show learning path as a table
                path_data = []
                for step in result["path"]:
                    step_num = step.get("step", path_data.__len__() + 1)
                    skill_type = "📋 Prerequisite" if step["is_prerequisite"] else "🎯 Target Skill"
                    path_data.append({
                        "Step": step_num,
                        "Skill": step["skill"].title(),
                        "Category": step["category"],
                        "Effort (days)": step["effort_days"],
                        "Score After": f"{step['score_after']}%",
                        "Type": skill_type
                    })

                st.table(path_data)

                # Algorithm trace
                with st.expander(f"🔬 {label} Algorithm Trace", expanded=False):
                    for t in result["trace"]:
                        st.markdown(f"**Step {t['step']}** — {t['action']}")
                        st.caption(t["description"])
                        st.markdown("---")


# ─── Tab 5: Job Route Optimizer (TSP) ────────────────────────────────────────
def render_tsp_tab():
    """Render the TSP Job Route Optimizer tab."""
    st.markdown("### 🗺️ Job Route Optimizer")
    st.markdown(
        "Uses the **Traveling Salesman Problem (TSP)** to find the optimal order "
        "for visiting job interview locations, minimizing total travel distance."
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Algorithm Explanation ──
    with st.expander("📖 How It Works — Traveling Salesman Problem", expanded=False):
        st.markdown("""
**The Traveling Salesman Problem (TSP)** is a classic NP-hard optimization problem:

> *Given a list of cities and the distances between them, find the shortest possible
> route that visits each city exactly once and returns to the starting city.*

**Algorithms Used:**

| Algorithm | Type | Complexity | When Used |
|-----------|------|------------|-----------|
| **Brute Force** | Exact (Optimal) | O(n!) | n ≤ 8 cities |
| **DP Held-Karp** | Exact (Optimal) | O(n² × 2ⁿ) | 8 < n ≤ 15 cities |
| **Nearest Neighbor** | Greedy Heuristic | O(n²) | All sizes |

**Mapping to Job Search:**
- **Cities** = Job interview locations
- **Distance** = Travel distance between cities (Haversine formula)
- **Tour** = Optimal order to attend all interviews
        """)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Job Input ──
    st.markdown("#### Enter Job Opportunities")
    st.markdown("Add job locations you're considering. We'll find the optimal travel route.")

    # Available cities for reference
    available_cities = sorted(CITY_COORDINATES.keys())

    num_jobs = st.slider("Number of job locations", min_value=3, max_value=10, value=5,
                         key="tsp_num_jobs")

    st.markdown("**Available cities:** " + ", ".join(
        [c.title() for c in available_cities[:20]]) + ", ...")

    # ── Home Location ──
    st.markdown("#### Starting Point (Your Home)")
    home_location = st.text_input("Home Location (City / Area / Address)", value="RS Puram, Coimbatore", key="tsp_home")

    st.markdown("#### Job Interview Locations")

    jobs = []
    
    if home_location.strip():
        coord = get_city_coordinates(home_location)
        if coord:
            jobs.append({
                "company": "🏠 Home",
                "city": home_location.strip(),
                "role": "Start/End Point"
            })
        else:
            st.warning(f"⚠️ Could not find coordinates for '{home_location}'. Route will not start here.")

    cols_per_row = 2

    for i in range(num_jobs):
        row = i // cols_per_row
        col_idx = i % cols_per_row

        if col_idx == 0:
            cols = st.columns(cols_per_row)

        with cols[col_idx]:
            st.markdown(f"**Job {i + 1}**")
            company = st.text_input(f"Company", value=f"Company {i + 1}",
                                    key=f"tsp_company_{i}")
            city_defaults = [
                "electronic city, bangalore", 
                "tidel park, chennai", 
                "saravanampatti, coimbatore", 
                "hitec city, hyderabad",
                "whitefield, bangalore",
                "omr, chennai"
            ]
            city = st.text_input(f"City / Area", value=city_defaults[i % len(city_defaults)] if i < 10 else "",
                                 key=f"tsp_city_{i}")
            role = st.text_input(f"Role", value="Software Engineer",
                                 key=f"tsp_role_{i}")

            if city.strip():
                coord = get_city_coordinates(city)
                if coord:
                    jobs.append({
                        "company": company.strip() or f"Company {i + 1}",
                        "city": city.strip(),
                        "role": role.strip() or "Engineer",
                    })
                else:
                    st.caption(f"⚠️ '{city}' not in database. Try another city.")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if st.button("🚀 Optimize Route", use_container_width=True, type="primary",
                 key="tsp_run_btn"):
        if len(jobs) < 3:
            st.warning("⚠️ Please enter at least 3 valid job locations.")
            return

        with st.spinner("🧠 Solving TSP with multiple algorithms..."):
            tsp_results = solve_tsp(jobs)
            st.session_state["tsp_results"] = tsp_results

    # ── Display Results ──
    if "tsp_results" in st.session_state and st.session_state["tsp_results"]:
        tsp_results = st.session_state["tsp_results"]

        if "error" in tsp_results:
            st.error(tsp_results["error"])
            return

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 TSP Results")

        n = tsp_results["num_jobs"]
        nn_result = tsp_results["nearest_neighbor"]
        exact_result = tsp_results.get("exact")
        comparison = tsp_results["comparison"]

        # ── Metrics ──
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cities", n)
        with col2:
            st.metric("Nearest Neighbor Distance", f"{nn_result['total_distance_km']} km")
        with col3:
            if exact_result:
                st.metric(
                    f"{tsp_results.get('exact_method', 'Exact')} Distance",
                    f"{exact_result['total_distance_km']} km"
                )
            else:
                st.metric("Exact Solution", "Skipped (too many cities)")

        # ── Comparison ──
        if exact_result:
            gap = comparison.get("optimality_gap_percent", 0)
            st.markdown(f"""
            <div class="comparison-box">
                <h4>🔍 Algorithm Comparison</h4>
                <p style="color: rgba(255,255,255,0.75); margin: 0;">
                    {comparison.get('summary', '')}
                    <br><br>
                    The Nearest Neighbor heuristic produced a tour that is
                    <strong>{gap}%</strong> longer than the optimal route.
                    {"This is quite close to optimal! ✅" if gap < 15 else "There's room for improvement — exact algorithms find better routes."}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Route Visualization ──
        st.markdown("### 🛣️ Optimal Route")

        if exact_result:
            tab_exact, tab_nn = st.tabs([
                f"🏆 {tsp_results.get('exact_method', 'Exact')} (Optimal)",
                "🔄 Nearest Neighbor (Heuristic)"
            ])
        else:
            tab_nn = st.container()
            tab_exact = None

        for tab, result, label in [(tab_exact, exact_result, "Exact"),
                                    (tab_nn, nn_result, "Nearest Neighbor")]:
            if tab is None or result is None:
                continue

            with tab:
                st.markdown(f"**Total Distance: {result['total_distance_km']} km**")
                st.markdown(f"**Route:** {' → '.join(result['route_names'])} → 🔄 Start")

                # Route segments table
                if "segments" in result:
                    seg_data = []
                    for idx, seg in enumerate(result["segments"], 1):
                        seg_data.append({
                            "Leg": idx,
                            "From": seg["from"],
                            "To": seg["to"],
                            "Distance (km)": seg["distance_km"]
                        })
                    st.table(seg_data)

                # Show on map — Interactive Folium with animated route path
                coords = tsp_results.get("coordinates", [])
                if coords:
                    import folium
                    from folium.plugins import AntPath
                    from streamlit_folium import st_folium

                    route_order = result.get("route", [])
                    job_names = tsp_results.get("job_names", [])
                    segments = result.get("segments", [])

                    # Build ordered route coordinates (lat, lon)
                    route_coords = []
                    for city_idx in route_order:
                        lat, lon = coords[city_idx]
                        if lat != 0 or lon != 0:
                            route_coords.append((lat, lon))

                    if route_coords:
                        st.markdown("**🗺️ Interactive Route Map**")

                        # Calculate map center
                        avg_lat = sum(c[0] for c in route_coords) / len(route_coords)
                        avg_lon = sum(c[1] for c in route_coords) / len(route_coords)

                        # Create Folium map
                        m = folium.Map(
                            location=[avg_lat, avg_lon],
                            zoom_start=7,
                            tiles="OpenStreetMap",
                        )

                        # Color palette for route segments
                        segment_colors = [
                            "#3b82f6", "#ef4444", "#10b981", "#f59e0b",
                            "#8b5cf6", "#ec4899", "#06b6d4", "#f97316",
                            "#14b8a6", "#6366f1",
                        ]

                        # Add numbered markers for each stop
                        for stop_idx, city_idx in enumerate(route_order):
                            lat, lon = coords[city_idx]
                            if lat == 0 and lon == 0:
                                continue
                            name = job_names[city_idx] if city_idx < len(job_names) else f"Stop {stop_idx}"

                            # Home marker (first stop) gets green; others blue
                            is_home = name.startswith("🏠")
                            icon_color = "green" if is_home else "blue"
                            icon_glyph = "home" if is_home else "briefcase"

                            # Build popup with segment distance
                            popup_text = f"<b>#{stop_idx + 1} — {name}</b>"
                            if stop_idx < len(segments):
                                seg = segments[stop_idx]
                                popup_text += f"<br>→ Next: {seg['to']}<br>Distance: {seg['distance_km']} km"

                            folium.Marker(
                                location=[lat, lon],
                                tooltip=f"#{stop_idx + 1} {name}",
                                popup=folium.Popup(popup_text, max_width=250),
                                icon=folium.Icon(
                                    color=icon_color,
                                    icon=icon_glyph,
                                    prefix="fa",
                                ),
                            ).add_to(m)

                        # Draw route segments as individual colored lines
                        for seg_idx in range(len(route_coords)):
                            from_coord = route_coords[seg_idx]
                            to_coord = route_coords[(seg_idx + 1) % len(route_coords)]
                            color = segment_colors[seg_idx % len(segment_colors)]

                            # Solid polyline for the segment
                            folium.PolyLine(
                                locations=[list(from_coord), list(to_coord)],
                                color=color,
                                weight=4,
                                opacity=0.8,
                                tooltip=f"Leg {seg_idx + 1}",
                            ).add_to(m)

                        # Animated ant-path overlay for the full route (including return)
                        full_path = [list(c) for c in route_coords]
                        full_path.append(list(route_coords[0]))  # close the loop
                        AntPath(
                            locations=full_path,
                            dash_array=[10, 20],
                            delay=1000,
                            weight=3,
                            color="#1e40af",
                            pulse_color="#60a5fa",
                            opacity=0.6,
                        ).add_to(m)

                        # Fit map bounds to show all markers
                        sw = [min(c[0] for c in route_coords) - 0.05,
                              min(c[1] for c in route_coords) - 0.05]
                        ne = [max(c[0] for c in route_coords) + 0.05,
                              max(c[1] for c in route_coords) + 0.05]
                        m.fit_bounds([sw, ne])

                        st_folium(m, use_container_width=True, height=500, returned_objects=[])

                # Algorithm trace
                with st.expander(f"🔬 {label} Algorithm Trace", expanded=False):
                    for t in result.get("trace", []):
                        desc = t.get("description", "")
                        action = t.get("action", "")
                        st.markdown(f"**Step {t.get('step', '?')}** — {action}")
                        st.caption(desc)
                        st.markdown("---")

        # Distance Matrix
        with st.expander("📏 Distance Matrix (km)", expanded=False):
            import pandas as pd
            dm = tsp_results["distance_matrix"]
            names = tsp_results["job_names"]
            short_names = [n.split("(")[0].strip() for n in names]
            df = pd.DataFrame(dm, index=short_names, columns=short_names)
            st.dataframe(df.style.format("{:.1f}"))


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Resume Upload",
        "🎯 JD Matching",
        "💬 Chatbot",
        "🧗 Career Path",
        "🗺️ Job Route"
    ])

    with tab1:
        render_upload_tab()

    with tab2:
        render_matching_tab()

    with tab3:
        render_chatbot_tab()

    with tab4:
        render_career_path_tab()

    with tab5:
        render_tsp_tab()


if __name__ == "__main__":
    main()
