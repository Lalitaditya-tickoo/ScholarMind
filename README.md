<div align="center">

# 🧠 ScholarMind

### AI-Powered Research Assistant with Quantified RAG Evaluation

**Upload papers → Ask questions → Get cited answers → Measure quality**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Claude API](https://img.shields.io/badge/Claude_API-Sonnet_4-D97757?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F61?style=for-the-badge)](https://www.trychroma.com)

<br>

*Not just another RAG demo — ScholarMind includes a **RAGAS-inspired evaluation pipeline**, **cross-paper synthesis**, and **table extraction** that quantify and differentiate the system from typical retrieval-augmented generation projects.*

---

</div>

## What Makes This Different

| Feature | Typical RAG Project | ScholarMind |
|---------|-------------------|-------------|
| **Query answering** | Upload → retrieve → answer | ✅ With inline source citations |
| **Evaluation** | "It works" | ✅ 4-metric RAGAS pipeline (Faithfulness, Relevance, Precision, Recall) |
| **Multi-paper analysis** | Query one paper at a time | ✅ Cross-paper synthesis: consensus, contradictions, research gaps |
| **Table extraction** | Ignores tables entirely | ✅ Structured table parsing with Markdown export |
| **Summarization** | Generic summary | ✅ Structured: Objective → Methods → Findings → Limitations |

---

## Architecture

```mermaid
graph TB
    subgraph Frontend ["⚛️ React Frontend"]
        UI[Chat Interface]
        PM[Paper Manager]
        CV[Citation Viewer]
    end

    subgraph API ["🚀 FastAPI Backend"]
        UP["/papers/upload"]
        QR["/query"]
        CP["/query/cross-paper"]
        SM["/papers/{id}/summarize"]
        TB["/papers/{id}/tables"]
        EV["/evaluate"]
    end

    subgraph Processing ["🔧 Processing Pipeline"]
        PDF[PDF Processor<br><i>PyMuPDF + OCR</i>]
        TE[Table Extractor<br><i>Layout Analysis</i>]
        CH[Chunker<br><i>512 tokens, 64 overlap</i>]
    end

    subgraph RAG ["🧠 RAG Engine"]
        EMB["Embedder<br><i>all-MiniLM-L6-v2</i>"]
        VS["Vector Store<br><i>ChromaDB (cosine)</i>"]
        RET[Retriever<br><i>Top-k Semantic Search</i>]
    end

    subgraph LLM ["🤖 Claude API"]
        ANS[Answer Generator<br><i>Cited responses</i>]
        SYN[Cross-Paper Synthesizer<br><i>Consensus + Contradictions</i>]
        SUM[Summarizer<br><i>Structured output</i>]
        JDG[Evaluation Judge<br><i>4-metric scoring</i>]
    end

    UI --> QR
    UI --> CP
    PM --> UP
    PM --> SM
    PM --> TB

    UP --> PDF --> CH --> EMB --> VS
    UP --> TE
    QR --> RET --> ANS
    CP --> RET --> SYN
    SM --> SUM
    EV --> RET --> JDG

    RET -.->|"query embedding"| EMB
    RET -.->|"similarity search"| VS

    style Frontend fill:#1a1a2e,stroke:#e94560,color:#fff
    style API fill:#16213e,stroke:#0f3460,color:#fff
    style Processing fill:#1a1a2e,stroke:#533483,color:#fff
    style RAG fill:#0f3460,stroke:#e94560,color:#fff
    style LLM fill:#533483,stroke:#e94560,color:#fff
```

---

## RAG Pipeline Deep Dive

```mermaid
sequenceDiagram
    participant U as User
    participant A as FastAPI
    participant P as PDF Processor
    participant R as RAG Engine
    participant V as ChromaDB
    participant C as Claude API

    Note over U,C: 📄 Paper Upload Flow
    U->>A: POST /papers/upload (PDF)
    A->>P: Extract text + tables
    P-->>A: pages[], tables[]
    A->>R: chunk_paper(text)
    R->>R: Split into 512-token chunks
    R->>R: Encode with MiniLM-L6-v2
    R->>V: Store embeddings + metadata
    V-->>A: chunk_count
    A-->>U: {paper_id, title, num_chunks}

    Note over U,C: 🔍 Query Flow
    U->>A: POST /query {question}
    A->>R: query(question, top_k=5)
    R->>R: Encode question
    R->>V: Cosine similarity search
    V-->>R: top-k chunks + scores
    R-->>A: RetrievedChunk[]
    A->>C: answer_query(question, chunks)
    C-->>A: Cited answer with [Source N]
    A-->>U: {answer, sources[], latency_ms}

    Note over U,C: 📊 Evaluation Flow
    U->>A: POST /evaluate {samples[]}
    loop For each sample
        A->>R: Retrieve chunks
        A->>C: Generate answer
        A->>C: Judge: Faithfulness
        A->>C: Judge: Relevance
        A->>C: Judge: Precision
        A->>C: Judge: Recall
    end
    A-->>U: {avg_scores, per_sample_results}
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/papers/upload` | Upload and process a PDF → returns metadata + chunk count |
| `GET` | `/papers` | List all uploaded papers with stats |
| `DELETE` | `/papers/{id}` | Remove paper and its vectors from the store |
| `POST` | `/query` | Ask a question → returns cited answer + sources |
| `POST` | `/query/cross-paper` | **Cross-paper synthesis** → consensus, contradictions, gaps |
| `POST` | `/papers/{id}/summarize` | Generate structured summary (Objective, Methods, Findings) |
| `GET` | `/papers/{id}/tables` | **Extract tables** from a paper's PDF |
| `POST` | `/evaluate` | **Run RAGAS-style evaluation** on the RAG pipeline |
| `GET` | `/health` | System health + vector store statistics |

<details>
<summary><b>📨 Example: Query Request & Response</b></summary>

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What preprocessing techniques improve underwater image quality?",
    "top_k": 5
  }'
```

```json
{
  "answer": "Several preprocessing techniques are used for underwater image enhancement. According to [Source 1], white balance correction using the gray world algorithm addresses color cast from selective light absorption. [Source 2] reports that CLAHE improves local contrast...",
  "sources": [
    {
      "paper_id": "a1b2c3d4",
      "paper_title": "Underwater Image Enhancement: A Survey",
      "chunk_index": 12,
      "page_number": 5,
      "relevance_score": 0.8934
    }
  ],
  "model": "claude-sonnet-4-20250514",
  "query_time_ms": 1847.3
}
```
</details>

<details>
<summary><b>🔀 Example: Cross-Paper Synthesis</b></summary>

```bash
curl -X POST http://localhost:8000/query/cross-paper \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How effective is transfer learning for underwater object detection?",
    "top_k_per_paper": 3
  }'
```

```json
{
  "consensus": "All papers agree that ImageNet-pretrained backbones significantly improve convergence speed and final accuracy on underwater datasets.",
  "contradictions": [
    {
      "topic": "Optimal backbone architecture",
      "paper_a": "Underwater Detection with CNNs",
      "position_a": "ResNet-101 provides the best accuracy-speed tradeoff",
      "paper_b": "Transformer-based Marine Detection",
      "position_b": "Swin Transformer outperforms all CNN backbones"
    }
  ],
  "gaps": [
    "No paper evaluates transfer learning from underwater-specific pretrained models",
    "Impact of domain gap between terrestrial and underwater imagery not quantified"
  ],
  "synthesis_summary": "While papers agree on the value of pretraining, they diverge on architecture choice, reflecting the broader CNN vs Transformer debate in the field."
}
```
</details>

---

## Evaluation Pipeline

ScholarMind doesn't just retrieve — it **measures how well it retrieves**. See [`EVALUATION.md`](./EVALUATION.md) for full methodology.

```mermaid
graph LR
    Q[Question + Ground Truth] --> R[1. Retrieve]
    R --> G[2. Generate Answer]
    G --> F{LLM Judge}
    F --> M1["Faithfulness<br><i>Hallucination detection</i>"]
    F --> M2["Answer Relevance<br><i>On-topic scoring</i>"]
    F --> M3["Context Precision<br><i>Retriever signal quality</i>"]
    F --> M4["Context Recall<br><i>Retriever coverage</i>"]
    M1 --> AGG[Aggregate Report]
    M2 --> AGG
    M3 --> AGG
    M4 --> AGG

    style F fill:#533483,stroke:#e94560,color:#fff
    style AGG fill:#0f3460,stroke:#e94560,color:#fff
```

### Metric Definitions

| Metric | Formula | What It Catches |
|--------|---------|----------------|
| **Faithfulness** | `supported_claims / total_claims` | Hallucinated facts not in sources |
| **Answer Relevance** | `semantic_similarity(answer, question)` | Off-topic or tangential responses |
| **Context Precision** | `relevant_chunks / retrieved_chunks` | Noisy retrieval (pulling irrelevant text) |
| **Context Recall** | `covered_facts / ground_truth_facts` | Missing information (incomplete retrieval) |

### Run Evaluation

```bash
cd eval
python run_eval.py \
  --dataset test_questions.json \
  --api http://localhost:8000 \
  --markdown RESULTS.md
```

---

## Project Structure

```
ScholarMind/
├── backend/
│   ├── main.py                # FastAPI server — all endpoints
│   ├── pdf_processor.py       # PDF extraction: text, OCR fallback
│   ├── rag_engine.py          # Chunking, embedding, ChromaDB, retrieval
│   ├── claude_client.py       # Claude API — answer generation + summarization
│   ├── cross_paper.py         # Cross-paper synthesis engine
│   ├── table_extractor.py     # Table detection + structured extraction
│   ├── evaluator.py           # RAGAS-inspired 4-metric evaluation pipeline
│   ├── models.py              # Pydantic request/response schemas
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main app: upload, chat, citations
│   │   └── main.jsx           # React entry point
│   ├── package.json
│   └── vite.config.js
├── eval/
│   ├── run_eval.py            # Standalone evaluation runner
│   └── test_questions.json    # Test dataset with ground truth answers
├── EVALUATION.md              # Evaluation methodology documentation
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Anthropic API Key](https://console.anthropic.com/)

### Backend

```bash
cd backend
pip install -r requirements.txt

# Create .env
echo ANTHROPIC_API_KEY=sk-ant-your-key-here > .env

# Start
uvicorn main:app --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Upload & Query

```bash
# Upload
curl -X POST -F "file=@paper.pdf" http://localhost:8000/papers/upload

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What methods does this paper propose?"}'

# Cross-paper (needs 2+ papers)
curl -X POST http://localhost:8000/query/cross-paper \
  -H "Content-Type: application/json" \
  -d '{"question": "How do these papers compare?"}'
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI | Async REST API with auto-generated docs |
| **Frontend** | React 18 + Vite | Chat interface with citation rendering |
| **Embedding** | sentence-transformers (`all-MiniLM-L6-v2`) | 384-dim vector encoding |
| **Vector Store** | ChromaDB | Persistent cosine similarity search |
| **LLM** | Claude Sonnet 4 (Anthropic API) | Generation, synthesis, evaluation |
| **PDF** | PyMuPDF | Text extraction, OCR, layout analysis |

---

## Design Decisions

**Why ChromaDB over Pinecone/Weaviate?**
Zero infrastructure — runs locally with file-based persistence. No API keys, no cloud dependency.

**Why custom evaluation over RAGAS library?**
Full control over judge prompts and scoring logic. No opaque dependencies. Each metric is a separate, debuggable LLM call.

**Why sentence-transformers over OpenAI embeddings?**
Runs locally (no API cost per embedding), reproducible, and `all-MiniLM-L6-v2` achieves strong STS benchmark performance while being fast enough for real-time.

**Why cross-paper synthesis?**
Real research isn't querying one paper — it's understanding how findings relate across literature. Finding contradictions between papers is genuinely hard and is what researchers actually need.

---

## License

MIT

---

<div align="center">

**Built by [Lalitaditya Tickoo](https://github.com/Lalitaditya-tickoo)**

*ScholarMind — Because research should be about understanding, not searching.*

</div>
