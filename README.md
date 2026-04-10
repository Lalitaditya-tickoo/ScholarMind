# ScholarMind — Multi-Modal RAG for Scientific Literature

> An AI-powered research assistant that ingests scientific papers (PDFs with text, figures, tables, equations) and lets you ask questions with cited, context-aware answers powered by Claude.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│  Upload PDFs → Ask Questions → Get Cited Answers         │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────────────┐
│                 FastAPI Backend                           │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ PDF Processor │  │  RAG Engine  │  │ Claude Client │  │
│  │              │  │              │  │               │  │
│  │ - Text Extract│  │ - Chunking   │  │ - Chat API    │  │
│  │ - Figure Extract│ │ - Embedding  │  │ - Vision API  │  │
│  │ - Table Detect│  │ - Retrieval  │  │ - Citations   │  │
│  │ - OCR Fallback│  │ - Re-ranking │  │               │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                 │                   │          │
│  ┌──────▼─────────────────▼───────────────────▼───────┐  │
│  │              ChromaDB (Vector Store)                │  │
│  │         Free • Local • No API Key Needed           │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| LLM | Claude API (Sonnet 4) | Pay-per-use (~$3/1M tokens) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | FREE (local) |
| Vector DB | ChromaDB | FREE (local) |
| PDF Processing | PyMuPDF + pdf2image | FREE |
| Backend | FastAPI (Python) | FREE |
| Frontend | React + Tailwind | FREE |

## Setup Instructions

### Step 1: Get Your API Keys

**Claude API Key (REQUIRED):**
1. Go to https://console.anthropic.com
2. Sign up / Log in
3. Go to API Keys → Create Key
4. Copy the key (starts with `sk-ant-...`)
5. Add $5 credits (minimum) — this will last you weeks of testing

### Step 2: Install Prerequisites

```bash
# Python 3.10+ required
python --version

# Node.js 18+ required  
node --version

# Install system dependencies (Ubuntu/Debian)
sudo apt install poppler-utils tesseract-ocr
```

### Step 3: Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env file
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env

# Run the server
uvicorn main:app --reload --port 8000
```

### Step 4: Frontend Setup

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

## Usage

1. Open the frontend at `http://localhost:5173`
2. Upload 1-5 scientific papers (PDF)
3. Wait for processing (text + figure extraction + embedding)
4. Ask questions like:
   - "What methodology did they use for data collection?"
   - "Compare the results across all uploaded papers"
   - "Explain Figure 3 from the second paper"
   - "What are the limitations mentioned?"
5. Get cited answers with page references

## Project Structure

```
multimodal-rag/
├── backend/
│   ├── main.py              # FastAPI server + endpoints
│   ├── pdf_processor.py     # PDF text/figure/table extraction
│   ├── rag_engine.py        # Chunking, embedding, retrieval
│   ├── claude_client.py     # Claude API integration
│   ├── models.py            # Pydantic models
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   │   └── App.jsx          # Main React app
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```
