<div align="center">

# 🇵🇰 Pakistan Budget 2026-27 — AI RAG Chatbot

### Ask anything about Pakistan's Federal Budget in English or Urdu

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pinecone](https://img.shields.io/badge/Pinecone-000024?style=for-the-badge&logo=pinecone&logoColor=white)](https://pinecone.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

![Stars](https://img.shields.io/github/stars/adeeljames/Pakistan-Budget-2026-27-RAG-Chatbot?style=social)
![Forks](https://img.shields.io/github/forks/adeeljames/Pakistan-Budget-2026-27-RAG-Chatbot?style=social)

<br />

<img src="https://img.shields.io/badge/Multi--Query%20Retrieval-✓-00611c?style=flat-square" />
<img src="https://img.shields.io/badge/Parent%20Document%20Retrieval-✓-00611c?style=flat-square" />
<img src="https://img.shields.io/badge/Bilingual%20(EN%2FUR)-✓-00611c?style=flat-square" />
<img src="https://img.shields.io/badge/Chat%20History%20Cache-✓-00611c?style=flat-square" />
<img src="https://img.shields.io/badge/Input%2FOutput%20Guardrails-✓-00611c?style=flat-square" />
<img src="https://img.shields.io/badge/461%20Vectors%20Indexed-✓-00611c?style=flat-square" />

</div>

---

## ✨ Overview

An **AI-powered RAG (Retrieval-Augmented Generation) chatbot** that answers questions about **Pakistan's Federal Budget 2026-2027**. It processes official budget documents and provides accurate, cited answers about expenditure, revenue, PSDP allocations, tax targets, development spending, and more — in both **English** and **Urdu**.

### Key Features

| Feature | Description |
|---------|-------------|
| 📄 **3 PDF Sources** | Annual Budget Statement (57 pages), Budget in Brief (50 pages), FM Speech Urdu (53 pages) |
| 🔍 **Multi-Query Retrieval** | Generates 4 query variants per question: specific English, keyword-augmented, Urdu translation, and mixed |
| 🧩 **Parent Document Retrieval** | Child chunks for precise search, parent chunks for full context |
| 🏆 **Score-based Reranking** | Ranks results by Pinecone similarity scores |
| 🤖 **Cerebras GLM-4.7** | Fast inference via Cerebras wafer-scale engine |
| 🛡️ **Guardrails** | Input topic filtering + output quality checks |
| ⚡ **Chat History Cache** | Instant answers for repeated questions via Pinecone qa_history namespace |
| 🌐 **Bilingual** | Full support for English and Urdu queries |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│ Input Guardrail │ ← Keyword-based topic filter (EN + UR)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ History Check   │ ← Pinecone qa_history (instant for repeats)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Multi-Query Gen │ ← 4 variants: specific, broad, Urdu, mixed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pinecone Search │ ← multilingual-e5-large embeddings (1024-dim)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Parent Doc Fetch│ ← Full context from local JSON store
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Score Reranking │ ← Sort by similarity, top-5 selection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Answer Gen      │ ← Cerebras GLM-4.7 with system prompt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Output Guardrail │ ← Quality heuristic check
└────────┬────────┘
         │
         ▼
   Clean Answer + Source Citation
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | [Cerebras](https://cerebras.ai/) — `zai-glm-4.7` (wafer-scale inference) |
| **Embeddings** | [Pinecone](https://pinecone.io/) — `multilingual-e5-large` (hosted, 1024-dim) |
| **Vector DB** | [Pinecone](https://pinecone.io/) — Serverless index (`budget-rag`) |
| **Orchestration** | [LangChain](https://langchain.com/) — Document loaders, text splitters |
| **UI** | [Streamlit](https://streamlit.io/) — Custom dark theme with glassmorphism |
| **Package Mgr** | [uv](https://github.com/astral-sh/uv) — Fast Python packaging |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Pinecone account with `multilingual-e5-large` index
- Cerebras API key

### 1. Clone & Setup

```bash
git clone https://github.com/adeeljames/Pakistan-Budget-2026-27-RAG-Chatbot.git
cd Pakistan-Budget-2026-27-RAG-Chatbot

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=budget-rag
PINECONE_HOST_URL=your_pinecone_host_url
CEREBRAS_API_KEY=your_cerebras_api_key
CEREBRAS_MODEL=zai-glm-4.7
EMBEDDING_MODEL=multilingual-e5-large
```

### 3. Ingest Documents (One-time)

```bash
python ingest.py
```

This will:
- Load all 3 PDFs (160 pages total)
- Create parent/child chunk hierarchy (169 parents → 461 children)
- Embed chunks via Pinecone Inference API
- Upsert to Pinecone vector index
- Save parent documents to `parent_store.json`

### 4. Run the Chatbot

**Terminal mode:**
```bash
python chatbot.py
```

**Streamlit UI:**
```bash
streamlit run app.py
```

---

## 💬 Sample Questions

| Category | English | Urdu |
|----------|---------|------|
| PSDP | What is the Federal PSDP allocation? | پی ایس ڈی پی کی مختص رقم کیا ہے؟ |
| Revenue | What are the green revenue components? | سبز آمدنی کے اجزاء کیا ہیں؟ |
| Expenditure | What is the current expenditure budget? | موجودہ اخراجات کا بجٹ کیا ہے؟ |
| Defence | What is the defence budget allocation? | دفاعی بجٹ کتنا ہے؟ |
| Grants | What are the grants and transfers? | گرانٹس اور ٹرانسفرز کیا ہیں؟ |
| Tax | What are the FBR tax targets? | ایف بی آر ٹیکس اہداف کیا ہیں؟ |

---

## 📂 Project Structure

```
Pakistan-Budget-2026-27-RAG-Chatbot/
├── app.py                      # Streamlit UI (beautiful dark theme)
├── chatbot.py                  # RAG pipeline + terminal chatbot
├── ingest.py                   # One-time PDF → Pinecone ingestion
├── config.py                   # Configuration (env vars + Streamlit secrets)
├── parent_store.json           # Parent document store (auto-generated)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── .env                        # API keys (not committed)
├── .gitignore                  # Git ignore rules
├── .streamlit/
│   ├── config.toml             # Streamlit theme configuration
│   └── secrets.toml.example    # Secrets template
└── docs/                       # Budget documents
    ├── Annual_Budget_Statement.pdf  # Official budget (57 pages)
    ├── Budget_in_Brief.pdf          # Budget summary (50 pages)
    └── FM_Speech_Urdu.pdf           # FM Speech Urdu (53 pages)
```

---

## 🎨 Streamlit UI

The app features a **custom dark theme** with:

- 🇵🇰 Pakistan flag green gradient hero header with shimmer animation
- 💎 Glassmorphism chat bubbles (user = right, bot = left)
- ⏳ Animated typing indicator
- 📋 Sidebar with clickable sample questions & pipeline stats
- 📎 Source citations in every answer
- ⚡ Instant cached answers for repeated questions

---

## 🔐 Streamlit Cloud Deployment

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Set main file to `app.py`
4. Add secrets in the app settings:

```toml
PINECONE_API_KEY = "your_key"
PINECONE_INDEX_NAME = "budget-rag"
PINECONE_HOST_URL = "your_host_url"
CEREBRAS_API_KEY = "your_key"
CEREBRAS_MODEL = "zai-glm-4.7"
EMBEDDING_MODEL = "multilingual-e5-large"
```

---

## 🧪 Retrieval Pipeline Details

| Step | Detail |
|------|--------|
| **Multi-Query** | 4 variants per question: original + keyword-augmented English + Urdu translation + mixed EN/UR |
| **Embeddings** | Microsoft `multilingual-e5-large` via Pinecone hosted inference (1024 dimensions) |
| **Chunking** | Parent: 2000 chars / 200 overlap → Child: 500 chars / 80 overlap |
| **Top-K** | 8 results per query variant, merged and deduplicated by parent_id |
| **Reranking** | Score-based sorting on Pinecone similarity scores |
| **History** | Q&A pairs stored in `qa_history` namespace, 0.92 similarity threshold for cache hits |
| **Guardrails** | Input: keyword topic filter (EN+UR) · Output: heuristic quality check |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### Made with ❤️ by [@muhammadadeelai](https://www.linkedin.com/in/muhammadadeelai/)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammadadeelai/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/adeeljames)

</div>
