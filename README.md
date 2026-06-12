# 📄 Resume Analyzer AI

An AI-powered Resume Analyzer that matches resumes against Job Descriptions and provides an interactive chatbot for resume Q&A — built with **LangChain**, **ChromaDB**, **HuggingFace**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference_API-yellow?logo=huggingface)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 **Multiple Resume Upload** | Upload multiple PDF resumes simultaneously, extract text, and index them in a unified vector database |
| 🎯 **JD Matching & Leaderboard** | Compare multiple resumes against a Job Description, ranking candidates on a real-time leaderboard with detailed gap analysis |
| 💬 **AI Multi-Resume Chatbot** | Ask questions about any or all resumes. RAG tracks the source file automatically so you can compare candidate credentials |
| 🧠 **Smart Analysis** | Identifies matching skills, gaps, and provides tailored improvement suggestions for each resume |
| 🔒 **Privacy-First** | Embeddings run locally; only LLM inference uses HuggingFace API |

---

## 🏗️ Architecture

```
User uploads PDF Resume
        │
        ▼
┌─────────────────────┐
│   PyPDF Extraction   │ ──► Raw text from PDF
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Text Cleaning &     │ ──► Clean, normalized text
│  Chunking (LangChain)│
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  HuggingFace         │ ──► Vector embeddings (runs locally!)
│  Embeddings          │
│  (all-MiniLM-L6-v2)  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ChromaDB            │ ──► Persistent vector storage
│  Vector Store        │
└─────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌─────────────────────┐  ┌─────────────────────┐
│  JD Matching Engine  │  │  RAG Chatbot         │
│  (LLM + Prompts)     │  │  (Retriever + LLM)   │
└─────────────────────┘  └─────────────────────┘
        │                          │
        ▼                          ▼
┌─────────────────────┐  ┌─────────────────────┐
│  Match Score +       │  │  Conversational Q&A  │
│  Skill Analysis      │  │  about Resume        │
└─────────────────────┘  └─────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Framework** | LangChain | AI orchestration, RAG pipeline |
| **Vector DB** | ChromaDB | Store & retrieve resume embeddings |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Convert text to vectors (FREE, local) |
| **LLM** | HuggingFace Inference API (Mistral-7B) | Text generation (FREE tier) |
| **PDF Parser** | PyPDF | Extract text from PDF resumes |
| **UI** | Streamlit | Web interface |
| **Env Management** | Python venv + python-dotenv | Isolated dependencies & secrets |

---

## 📋 Prerequisites

Before you begin, make sure you have:

- ✅ **Python 3.9 or higher** installed ([Download Python](https://www.python.org/downloads/))
- ✅ **pip** (comes with Python)
- ✅ **Git** (optional, for cloning)
- ✅ **HuggingFace Account** (free) — [Sign up here](https://huggingface.co/join)

---

## 🚀 Step-by-Step Setup Guide

### Step 1: Clone or Navigate to the Project

```bash
cd c:\project\resumeanalyzer
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
# source venv/bin/activate
```

> 💡 You should see `(venv)` at the beginning of your terminal prompt after activation.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This may take a few minutes as it downloads LangChain, ChromaDB, sentence-transformers, etc.

### Step 4: Set Up Your HuggingFace Token

1. Go to [HuggingFace Token Settings](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Give it a name (e.g., "resume-analyzer")
4. Select **"Read"** access
5. Copy the token (starts with `hf_...`)

Now edit the `.env` file:

```bash
# Open .env in your editor and replace the placeholder:
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501** 🎉

---

## 📖 How to Use

### 1. Upload Resumes (Tab 1)
- Click the **"📄 Resume Upload"** tab
- Click **"Browse files"** and select one or multiple PDF resumes
- Wait for processing (text extraction → chunking with source metadata → embedding → vector storage)
- View database stats and choose a resume from the dropdown to inspect the extracted text
- Delete/add resumes at any time; the database will auto-rebuild dynamically

### 2. Match with Job Description & Compare (Tab 2)
- Click the **"🎯 JD Matching"** tab
- Paste the full Job Description in the text area
- Click **"🔍 Analyze & Compare All Resumes"**
- View the **🏆 Candidate Leaderboard** showing all resumes sorted by their match score (0-100)
- Select any candidate from the dropdown to read their specific detailed analysis, skill matches, gaps, and improvements

### 3. Chat with AI (Tab 3)
- Click the **"💬 Chatbot"** tab
- Ask any question about the uploaded resumes (e.g. comparing profiles, finding specific candidates, querying skills)
- The assistant is context-aware and knows which candidate matches which information via document source tags!
- Example questions:
  - *"Which candidate has experience with Django?"*
  - *"Compare the experience levels of Alice and Bob"*
  - *"List the core strengths of candidate John"*
  - *"Who would be the best fit for a DevOps role based on their resume?"*

---

## 📁 Project Structure

```
resumeanalyzer/
├── .env                    # Your HuggingFace token (gitignored)
├── .env.example            # Template for .env
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── app.py                  # Main Streamlit application
├── src/
│   ├── __init__.py         # Package init
│   ├── resume_parser.py    # PDF extraction & text chunking
│   ├── embeddings.py       # HuggingFace embeddings & ChromaDB
│   ├── jd_matcher.py       # JD matching & scoring logic
│   └── chatbot.py          # RAG chatbot with LangChain
├── prompts/
│   ├── __init__.py         # Package init
│   ├── matching_prompt.py  # Prompt templates for JD matching
│   └── chatbot_prompt.py   # Prompt templates for chatbot
└── chroma_db/              # ChromaDB storage (auto-created, gitignored)
```

---

## 🔧 Troubleshooting

### ❌ "LLM Not Connected"
- Make sure you've added your HuggingFace token to `.env`
- The token should start with `hf_`
- Restart the Streamlit app after updating `.env`

### ❌ "Model is loading" errors
- HuggingFace free tier models may take 20-30 seconds to cold-start
- Simply try again after waiting a moment

### ❌ Slow embedding creation
- The first time embeddings run, the model (~90MB) is downloaded
- Subsequent runs use the cached model and are faster

### ❌ ChromaDB errors
- Delete the `chroma_db/` folder and re-upload your resume
- Make sure you have write permissions in the project directory

### ❌ Import errors
- Make sure your virtual environment is activated: `venv\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`

---

## 🔄 Switching LLM Providers

The app is designed to easily swap LLM providers. Edit `app.py`'s `initialize_llm()` function:

### Use Google Gemini (Free)
```bash
pip install langchain-google-genai
```
```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="YOUR_KEY")
```

### Use Ollama (Fully Local, No API Key)
```bash
pip install langchain-ollama
# Also install Ollama: https://ollama.com
ollama pull mistral
```
```python
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="mistral")
```

### Use Groq (Fast, Free Tier)
```bash
pip install langchain-groq
```
```python
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key="YOUR_KEY")
```

---

## 📄 License

This project is for educational purposes. Feel free to modify and use it.

---

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) — AI framework
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [HuggingFace](https://huggingface.co/) — Models & inference
- [Streamlit](https://streamlit.io/) — UI framework
- [Sentence Transformers](https://www.sbert.net/) — Embedding models
