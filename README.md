# Month 9 — NLP + Deep Learning Portfolio
### Part of a 12-Month Data Science & AI Self-Study Program

![Score](https://img.shields.io/badge/Score-1150%2F1150-brightgreen?style=flat-square)
![Stars](https://img.shields.io/badge/Bonus-140★-gold?style=flat-square)
![Days](https://img.shields.io/badge/Days-155--168-blue?style=flat-square)
![Environment](https://img.shields.io/badge/Environment-Google%20Colab-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## Overview

Month 9 covers the full NLP and Deep Learning stack — from neural network fundamentals through transformer fine-tuning to a production-grade RAG system. Every notebook runs on **Google Colab (T4 GPU)** and uses a single consistent dataset: **ReviewPulse India** (600 rows, seed=155), a synthetic freelancer review dataset with sentiment, rating, and re-hire labels.

The month closes with a complete RAG pipeline: LangChain + FAISS + Groq (Llama 3.1) + RAGAS evaluation — the same architecture used in real AI-powered analytics products.

---

## Scorecard

| Day | Topic | Score |
|-----|-------|-------|
| 155 | Neural Networks & Keras | 90/90 + 10★ |
| 156 | Convolutional Neural Networks (CNNs) | 80/80 + 10★ |
| 157 | RNNs & LSTMs | 80/80 + 10★ |
| 158 | NLP Text Processing | 80/80 + 10★ |
| 159 | HuggingFace Transformers (Zero-Shot) | 80/80 + 10★ |
| 160 | RAG Pipeline (Foundations) | 80/80 + 10★ |
| 161 | Word Embeddings (Word2Vec + GloVe) | 90/90 + 10★ |
| 162 | Sentence Embeddings & Semantic Search | 80/80 + 10★ |
| 163 | Topic Modeling with LDA | 80/80 + 10★ |
| 164 | DistilBERT Fine-Tuning (Binary) | 90/90 + 10★ |
| 165 | Fine-Tuning Part 2 — Save / Load / Inference | 80/80 + 10★ |
| 166 | compute_metrics + TrainingArguments + EarlyStoppingCallback | 80/80 + 10★ |
| 167 | Multi-Class Fine-Tuning (3-Class Sentiment) | 80/80 + 10★ |
| 168 | RAG + LangChain + Groq Capstone | 80/80 + 10★ |
| **Total** | | **1150/1150 + 140★** |

**14 days. 14 perfect scores.**

---

## Dataset — ReviewPulse India

All notebooks share one dataset, generated deterministically at the top of each notebook.

```python
np.random.seed(155)
n = 600
# Fields: freelancer_id, review_text, rating (1-5), sentiment, hired_again
# Distribution: negative=267 (44.5%), neutral=180 (30.0%), positive=153 (25.5%)
```

| Field | Type | Description |
|-------|------|-------------|
| `freelancer_id` | string | FL0001 – FL0600 |
| `review_text` | string | Synthetic review (~56 chars avg) |
| `rating` | int | 1–5, correlated with sentiment |
| `sentiment` | category | negative / neutral / positive |
| `hired_again` | binary | 1 = re-hired; positive cohort = 85% re-hire rate |

---

## Notebooks

### Week 1 — Deep Learning Foundations

**Day 155 — Neural Networks & Keras**
- Built a feedforward neural network on TF-IDF features
- Dropout regularization, BatchNormalization, learning rate scheduling
- Compared MLP vs Logistic Regression baseline on ReviewPulse

**Day 156 — CNNs**
- Applied 1D Convolutional layers to text sequences
- Filter sizes, MaxPooling, GlobalMaxPooling for text classification
- Conv1D sentiment classifier vs MLP baseline

**Day 157 — RNNs & LSTMs**
- Sequence modeling with SimpleRNN → LSTM → Bidirectional LSTM
- Embedding layer + LSTM for sentiment classification
- Vanishing gradient analysis and why LSTM gates solve it

**Day 158 — NLP Text Processing**
- Full preprocessing pipeline: tokenization, stopword removal, lemmatization
- TF-IDF with unigrams + bigrams, vocabulary analysis
- Text feature engineering: review length, punctuation density, word diversity

---

### Week 2 — Embeddings & Transformers

**Day 159 — HuggingFace Transformers (Zero-Shot)**
- DistilBERT sentiment pipeline — zero-shot inference on ReviewPulse
- CLS token embedding extraction (768-dim)
- Logistic regression trained on CLS embeddings vs TF-IDF baseline

**Day 160 — RAG Pipeline (Foundations)**
- Document loading, chunking strategy, vector store fundamentals
- FAISS IndexFlatL2 from scratch — no LangChain abstraction
- Similarity search and retrieval mechanics

**Day 161 — Word Embeddings (Word2Vec + GloVe)**
- Word2Vec (CBOW + Skip-gram) trained on ReviewPulse corpus
- GloVe pretrained vectors loaded and queried
- Analogy tasks, nearest-neighbour analysis, embedding visualization (t-SNE)

**Day 162 — Sentence Embeddings & Semantic Search**
- all-MiniLM-L6-v2 (384-dim) via sentence-transformers
- Cosine similarity matrix across all 600 reviews
- Semantic search: query → top-k most similar reviews

---

### Week 3 — Fine-Tuning DistilBERT

**Day 163 — Topic Modeling with LDA**
- CountVectorizer → Document-Term Matrix → LDA (n_components=4)
- Top-10 words per topic extraction and business naming
- Topic-to-hired_again hire rate analysis
- Perplexity comparison across k=3,4,5

**Day 164 — DistilBERT Fine-Tuning (Binary)**
- Full fine-tuning loop: tokenization → Dataset → DataCollator → Trainer
- Binary sentiment (negative vs positive), stratified 70/15/15 split
- Per-epoch F1 tracking, best model checkpoint saving

**Day 165 — Save / Load / Inference**
- `save_pretrained()` + `from_pretrained()` workflow
- Inference pipeline: single text → label + confidence
- Batch prediction with confidence thresholding
- Model portability: loading from local path vs HuggingFace Hub

**Day 166 — compute_metrics + TrainingArguments + Callbacks**
- `compute_metrics` with precision, recall, F1 (macro + weighted)
- `TrainingArguments` deep-dive: warmup_steps, weight_decay, eval_strategy
- `EarlyStoppingCallback` with patience=2
- Epoch-by-epoch log table: loss, eval_loss, F1 per epoch
- Baseline NRA comparison: fine-tuned vs zero-shot pipeline

**Day 167 — Multi-Class Fine-Tuning (3-Class)**
- Extended to 3-class sentiment: negative / neutral / positive
- `num_labels=3`, label encoding, per-class classification report
- Confusion matrix heatmap with business interpretation
- Deployment inference function: `predict(text)` → label + confidence + all_probs
- PEFT/LoRA awareness: comparison table vs full fine-tuning

---

### Week 4 — RAG Capstone

**Day 168 — RAG + LangChain + Groq Capstone**

> *End-to-end production QA system: natural language queries over 600 client reviews — no SQL required.*

**Architecture:**
```
ReviewPulse Reviews
       │
   Document Loader → RecursiveCharacterTextSplitter (chunk_size=200)
       │
   all-MiniLM-L6-v2 Embeddings (384-dim)
       │
   FAISS Vector Store (600 vectors)
       │
   Retriever (k=3)
       │
   Prompt Template (grounded — context only)
       │
   Groq LLM (llama-3.1-8b-instant, temp=0)
       │
   Answer + RAGAS Evaluation
```

**Tasks completed:**
- T1: LangChain `Document` objects (600 docs, 4 metadata fields) + text splitting
- T2: FAISS vector store (384-dim, 600 vectors) + similarity search verification
- T3: Full RAG chain via LangChain LCEL Runnable API + 4 business queries
- T4: Metadata filtering (`filter={'sentiment': 'negative'}`) + MMR retrieval (`lambda_mult=0.5`)
- T5: Manual RAGAS evaluation (faithfulness, answer_relevancy, context_precision, context_recall)
- Bonus: FAISS persist → reload → verify + 4-sentence client business summary

**RAGAS Results:**

| Metric | Score |
|--------|-------|
| Faithfulness | 0.375 |
| Answer Relevancy | 0.650 |
| Context Precision | 0.425 |
| Context Recall | 0.156 ← weakest |

Context recall is lowest because k=3 short reviews cannot cover dataset-level aggregate statistics required by ground-truth answers. Fix: increase k to 10 or add a corpus summarisation layer.

**Stack:** `langchain` · `langchain-groq` · `sentence-transformers` · `faiss-cpu` · `ragas` · `datasets`

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Environment | Google Colab (T4 GPU) |
| Deep Learning | TensorFlow/Keras, PyTorch |
| NLP | HuggingFace Transformers, sentence-transformers, Gensim |
| Fine-Tuning | DistilBERT (`distilbert-base-uncased`), PEFT/LoRA (awareness) |
| RAG | LangChain (LCEL), FAISS, Groq API |
| LLM | llama-3.1-8b-instant via Groq (free tier) |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| Evaluation | RAGAS (faithfulness, relevancy, precision, recall) |
| Analysis | scikit-learn, pandas, numpy, matplotlib, seaborn |

---

## Key Concepts Covered

**Deep Learning**
- Forward pass, backpropagation, gradient descent
- Dropout, BatchNorm, learning rate scheduling
- Sequence models: RNN → LSTM → Bidirectional LSTM
- 1D CNN for text, filter sizes, pooling

**NLP**
- Tokenization, lemmatization, stopword removal
- TF-IDF (unigram + bigram), vocabulary analysis
- Word2Vec (CBOW + Skip-gram), GloVe pretrained vectors
- Sentence embeddings, cosine similarity, semantic search
- Topic modeling (LDA), document-term matrix

**Transformers & Fine-Tuning**
- BERT/DistilBERT architecture: CLS token, attention, tokenizer
- Full fine-tuning pipeline: Dataset → DataCollator → Trainer
- compute_metrics, TrainingArguments, EarlyStoppingCallback
- Binary and multi-class classification with DistilBERT
- Save/load/inference workflow for deployment

**RAG & LangChain**
- Document loading, chunking, embedding, indexing
- FAISS vector store: IndexFlatL2, similarity search, MMR
- Metadata filtering in retrieval
- LangChain LCEL: Runnable chains, PromptTemplate, StrOutputParser
- RAG evaluation: RAGAS 4-metric framework

---

## NRA Framework

Every analytical output in this month follows the **NRA (Number → Reason → Action)** discipline:

- **Number** — exact value read from printed cell output, never estimated
- **Reason** — causal mechanism explaining *why* the number is what it is (not an outcome description)
- **Action** — specific, committed next step (named model/threshold/parameter — no hedging)

Example from Day 168 T5:
> *Number: context_recall = 0.156. Reason: k=3 short reviews (avg 56 chars) cannot cover dataset-level aggregate statistics like "44.5% negative" or "85% rehire rate" required by ground-truth answers — the retrieved context is too narrow. Action: Increase k to 10 and add a corpus summarisation layer for questions requiring global dataset insights.*

---

## Running the Notebooks

All notebooks are self-contained. Run in order within each notebook — no external files needed.

```
1. Open any .ipynb in Google Colab
2. Runtime → Change runtime type → T4 GPU (recommended for Days 164–167)
3. Run all cells in sequence
4. For Day 168: add GROQ_API_KEY to Colab Secrets (Runtime → Secrets)
```

**Groq API** (Day 168): Free tier at [console.groq.com](https://console.groq.com) — no credit card required.

---

## Part of a Larger Program

| Month | Focus | Status |
|-------|-------|--------|
| M1 | Excel & Data Analytics | ✅ Complete |
| M2 | SQL | ✅ Complete |
| M3 | Python & Pandas | ✅ Complete |
| M4 | Power BI & Tableau | ✅ Complete |
| M5 | BI + Upwork Launch | ✅ Complete |
| M6 | Statistics & ML | ✅ Complete |
| M7 | Advanced ML | ✅ Complete |
| M8 | Streamlit & FastAPI | ✅ Complete |
| **M9** | **NLP + Deep Learning** | ✅ **Complete — 1150/1150 + 140★** |
| M10 | LangChain + MLflow + Evidently | 🔄 Next |
| M11 | Data Engineering + CI/CD + AWS | ⏳ Upcoming |
| M12 | Capstone + Netherlands MSc Applications | ⏳ Upcoming |

---

## Author

**Deepanshu** · [@deepanshu0110](https://github.com/deepanshu0110)

*Targeting: TU/e MSc Data Science & AI (Sep 2027) · Freelance AI/Data Analyst · Databrief product*
