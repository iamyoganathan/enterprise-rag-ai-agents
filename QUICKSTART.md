# 🚀 Quick Start Guide - Enterprise RAG System

## Step 1: Environment Setup

### Create and activate virtual environment:

```bash
# Navigate to project directory
cd enterprise-rag-ai-agents

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Linux/Mac:
# source venv/bin/activate
```

## Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**This will install:**
- LangChain & LangGraph for LLM orchestration
- FastAPI for REST API
- Streamlit for UI
- ChromaDB for vector storage
- sentence-transformers for embeddings
- All other dependencies (~60+ packages)

**Installation time:** 5-10 minutes depending on internet speed

## Step 3: Configure Environment Variables

```bash
# Copy the example environment file
copy .env.example .env

# Edit .env file and add your API keys
# Windows:
notepad .env

# Linux/Mac:
# nano .env
```

**Required API Keys (FREE):**

1. **Groq API (Recommended - Fast & Free):**
   - Visit: https://console.groq.com
   - Sign up and create API key
   - Add to .env: `GROQ_API_KEY=your_key_here`

2. **OpenAI API (Optional):**
   - Visit: https://platform.openai.com
   - Add to .env: `OPENAI_API_KEY=your_key_here`

3. **LangSmith (Optional - for monitoring):**
   - Visit: https://smith.langchain.com
   - Add to .env: `LANGCHAIN_API_KEY=your_key_here`

**Minimum .env configuration:**
```
GROQ_API_KEY=your_groq_api_key_here
DEFAULT_LLM_PROVIDER=groq
DEFAULT_LLM_MODEL=llama3-70b-8192
```

## Step 4: Test Installation

```bash
# Test configuration
python -c "from src.utils.config import get_settings; print('Config loaded successfully!')"

# Test imports
python -c "import langchain; import chromadb; import fastapi; import streamlit; print('All imports successful!')"
```

## Step 5: Initialize the System

```bash
# Create necessary directories
python -c "from src.utils.config import get_settings; get_settings().ensure_directories(); print('Directories created!')"
```

## Step 6: Run the Application

### Option A: Streamlit UI (Recommended for beginners)

```bash
streamlit run src/frontend/app.py
```

Then open: http://localhost:8501

### Option B: FastAPI Server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open API docs: http://localhost:8000/docs

## Step 7: Add Sample Documents

1. Place PDF/DOCX/TXT files in `data/sample_documents/`
2. Use the UI or API to upload and process them
3. Start asking questions!

## Common Issues & Solutions

### Issue 1: Module not found error
```bash
# Solution: Make sure you're in the right directory and venv is activated
cd enterprise-rag-ai-agents
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Issue 2: Port already in use
```bash
# Solution: Use different port
streamlit run src/frontend/app.py --server.port 8502
# Or for API:
uvicorn src.api.main:app --port 8001
```

### Issue 3: API key not found
```bash
# Solution: Make sure .env file exists and has correct keys
# Check if .env is in the root directory (not in src/)
dir .env  # Windows
# ls -la .env  # Linux/Mac
```

### Issue 4: ChromaDB initialization error
```bash
# Solution: Clear and reinitialize
Remove-Item -Recurse -Force data\vector_db  # Windows
# rm -rf data/vector_db  # Linux/Mac
```

## Next Steps

1. **Phase 1 (Week 1-2):** Document Ingestion
   - Implement document loaders
   - Test with sample PDFs
   - Verify chunking works

2. **Phase 2 (Week 3-4):** RAG Pipeline
   - Implement retrieval
   - Integrate LLM
   - Test Q&A functionality

3. **Phase 3 (Week 5-6):** AI Agents
   - Build agent system
   - Add complex query handling

4. **Phase 4 (Week 7-8):** Polish & Deploy
   - Evaluation metrics
   - Docker deployment
   - Documentation

## Useful Commands

```bash
# Check Python version
python --version  # Should be 3.10+

# List installed packages
pip list

# Test specific module
python src/utils/config.py
python src/utils/logger.py
python src/utils/cache.py

# Run tests (once written)
pytest tests/

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## Development Workflow

1. **Daily:**
   - Activate venv
   - Pull latest changes (if using git)
   - Work on assigned module
   - Test your changes
   - Commit progress

2. **Weekly:**
   - Review completed modules
   - Update documentation
   - Run integration tests
   - Deploy to test environment

## Resources

- **LangChain Docs:** https://python.langchain.com/
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Streamlit Docs:** https://docs.streamlit.io/
- **ChromaDB Docs:** https://docs.trychroma.com/

## Getting Help

1. Check the docs/ directory for detailed documentation
2. Review example code in notebooks/
3. Check the PROJECT_PROPOSAL.md for architecture details
4. Search existing issues on GitHub

---

## Quick Test Script

Save this as `test_setup.py` and run it:

```python
"""Test if everything is set up correctly."""

def test_setup():
    print("🔍 Testing setup...\n")
    
    # Test 1: Python version
    import sys
    print(f"✅ Python version: {sys.version}")
    
    # Test 2: Import core packages
    try:
        import langchain
        print("✅ LangChain installed")
    except:
        print("❌ LangChain not installed")
    
    try:
        import chromadb
        print("✅ ChromaDB installed")
    except:
        print("❌ ChromaDB not installed")
    
    try:
        import fastapi
        print("✅ FastAPI installed")
    except:
        print("❌ FastAPI not installed")
    
    try:
        import streamlit
        print("✅ Streamlit installed")
    except:
        print("❌ Streamlit not installed")
    
    # Test 3: Configuration
    try:
        from src.utils.config import get_settings
        settings = get_settings()
        print(f"✅ Configuration loaded")
        print(f"   LLM Provider: {settings.default_llm_provider}")
        print(f"   LLM Model: {settings.default_llm_model}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
    
    # Test 4: Logger
    try:
        from src.utils.logger import log
        log.info("Test log message")
        print("✅ Logger working")
    except Exception as e:
        print(f"❌ Logger error: {e}")
    
    print("\n✨ Setup test complete!")

if __name__ == "__main__":
    test_setup()
```

Run it:
```bash
python test_setup.py
```

---

**🎉 You're ready to start building! Begin with Module 1: Document Ingestion**
