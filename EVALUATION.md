# Evaluation Methodology

ScholarMind includes a custom RAGAS-inspired evaluation pipeline that quantifies RAG pipeline quality using **LLM-as-judge** scoring across 4 key metrics.

## Why Evaluate?

Most RAG demos stop at "it returns answers." ScholarMind goes further by measuring **how good** those answers are — the same way production AI systems are validated.

## Metrics

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Faithfulness** | Are all claims in the answer grounded in retrieved context? | Detects hallucination — the #1 failure mode in RAG |
| **Answer Relevance** | Does the answer address the actual question asked? | Catches tangential or off-topic responses |
| **Context Precision** | Are the retrieved chunks relevant to the question? | Measures retriever quality |
| **Context Recall** | Does the retrieved context contain enough info for a correct answer? | Measures retriever coverage |

## How It Works

```
Question + Ground Truth
        |
        v
  1. Retrieve    --> ChromaDB returns top-k chunks
  2. Generate    --> Claude generates cited answer
  3. Judge       --> Claude (separate call) scores each metric
        |
        v
   Aggregate Report
```

## Running Evaluation

```bash
# 1. Start backend with papers uploaded
cd backend && uvicorn main:app --port 8000

# 2. Upload test papers
curl -X POST -F "file=@paper1.pdf" http://localhost:8000/papers/upload

# 3. Run evaluation
cd ../eval
python run_eval.py --dataset test_questions.json --api http://localhost:8000 --markdown RESULTS.md
```

## Design Decisions

- **LLM-as-judge over string matching**: Semantic evaluation captures meaning, not just word overlap
- **Custom implementation over RAGAS library**: Full control over prompts, scoring, and debugging
- **4 orthogonal metrics**: Each measures a different failure mode
