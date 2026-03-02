# 🚀 Enterprise RAG System with AI Agent Orchestration

> **An intelligent document analysis system powered by Retrieval Augmented Generation (RAG) and autonomous AI agents**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-Latest-orange)](https://python.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green)](https://fastapi.tiangolo.com/)

## 📋 Overview

This project implements a production-ready RAG system that combines:
- **Document Intelligence**: Process PDFs, DOCX, TXT, and Markdown files
- **Semantic Search**: Vector embeddings with ChromaDB/FAISS
- **AI Agents**: Multi-agent orchestration for complex queries
- **LLM Integration**: Support for OpenAI, Groq, and HuggingFace models
- **Production Features**: REST API, web UI, caching, monitoring, evaluation

## ✨ Features

### Core Capabilities
- 📄 **Multi-format document support** (PDF, DOCX, TXT, MD)
- 🔍 **Advanced retrieval** with semantic search and re-ranking
- 🤖 **AI Agent system** for multi-step reasoning
- 💬 **LLM-powered answers** with source citations
- 📊 **Comprehensive evaluation** using RAGAS metrics
- 🌐 **REST API** with authentication and rate limiting
- 🎨 **Interactive web UI** built with Streamlit
- 💾 **Response caching** for performance optimization
- 📈 **Monitoring** with LangSmith integration

### Advanced Features
- Hybrid search (keyword + semantic)
- Query expansion and reformulation
- Cross-encoder re-ranking
- Context-aware chunking
- Multi-agent collaboration
- Conversation memory management
- A/B testing framework
- Cost tracking and optimization

## 🏗️ Architecture

```
User Interface (Streamlit)
         ↓
    REST API (FastAPI)
         ↓
   Agent Orchestrator
    ↙️    ↓    ↘️
Search  Analysis  Synthesis
Agent   Agent     Agent
    ↘️    ↓    ↙️
   Retrieval Engine
         ↓
   Vector Database
         ↓
Document Ingestion
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | LangChain, LangGraph |
| **LLMs** | OpenAI, Groq, HuggingFace |
| **Embeddings** | sentence-transformers |
| **Vector DB** | ChromaDB, FAISS |
| **API** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Evaluation** | RAGAS |
| **Monitoring** | LangSmith, Loguru |

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Git
- (Optional) Docker

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/enterprise-rag-ai-agents.git
cd enterprise-rag-ai-agents
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# Minimum required: GROQ_API_KEY (free from groq.com)
```

5. **Initialize the system**
```bash
python scripts/setup_vectordb.py
```

## 🚀 Usage

### Option 1: Run with Streamlit UI

```bash
streamlit run src/frontend/app.py
```

Open your browser at `http://localhost:8501`

### Option 2: Run API Server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`

### Option 3: Docker

```bash
docker-compose up
```

## 📚 Project Structure

```
enterprise-rag-ai-agents/
├── src/
│   ├── ingestion/          # Document loading & processing
│   ├── embeddings/         # Vector embeddings
│   ├── retrieval/          # Semantic search & retrieval
│   ├── agents/             # AI agent orchestration
│   ├── llm/                # LLM integration
│   ├── evaluation/         # Metrics & benchmarking
│   ├── api/                # FastAPI endpoints
│   ├── frontend/           # Streamlit UI
│   └── utils/              # Shared utilities
├── data/                   # Data storage
├── tests/                  # Unit & integration tests
├── notebooks/              # Jupyter notebooks
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── docker/                 # Docker files
└── docs/                   # Documentation
```

## 🔧 Configuration

Edit `configs/model_config.yaml` to customize:
- Model selection and parameters
- Chunking strategy
- Retrieval settings
- Agent behavior
- Evaluation metrics

## 📖 Documentation

- [Architecture Documentation](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)

## 🧪 Testing

Run tests:
```bash
# All tests
pytest

# With coverage
pytest --cov=src tests/

# Specific module
pytest tests/test_retrieval.py
```

## 📊 Evaluation

Run evaluation suite:
```bash
python scripts/evaluate_system.py
```

Metrics include:
- Faithfulness
- Answer Relevance
- Context Precision & Recall
- Response latency
- Cost per query

## 🎯 Getting API Keys (All FREE!)

### 1. Groq API (Recommended - Fastest & Free)
- Visit: https://console.groq.com
- Sign up and get API key
- Free tier: 30 requests/minute

### 2. OpenAI (Optional)
- Visit: https://platform.openai.com
- Sign up and get API key
- Free tier: $5 credit

### 3. HuggingFace (Optional)
- Visit: https://huggingface.co/settings/tokens
- Create account and generate token
- Completely free

### 4. LangSmith (Optional - Monitoring)
- Visit: https://smith.langchain.com
- Sign up and get API key
- Free tier: 1000 traces/month

## 🚢 Deployment

### Deploy to Render (Free)
```bash
# API deployment
1. Connect GitHub repo to Render
2. Select "Web Service"
3. Build command: pip install -r requirements.txt
4. Start command: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

### Deploy to Streamlit Cloud (Free)
```bash
1. Push code to GitHub
2. Visit share.streamlit.io
3. Connect repo and select src/frontend/app.py
4. Add secrets (API keys) in settings
```

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) first.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 🎓 Academic Use

This project was developed as part of an MCA final year project. Feel free to use it for:
- Academic projects
- Learning and experimentation
- Portfolio demonstrations
- Research purposes

Please cite if used in academic work.

## 🌟 Acknowledgments

- LangChain team for the amazing framework
- HuggingFace for open-source models
- Groq for providing free LLM inference
- All open-source contributors

## 📧 Contact

- **Author**: Yoganathan C
- **Email**: iamyoganathanc@gmail.com
- **LinkedIn**: [@Yoganathan](www.linkedin.com/in/yoganathan-c)
- **GitHub**: [@iamyoganathan](https://github.com/iamyoganathan)

---

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for the AI community
