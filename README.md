# 🧠 RAG-Based Semantic Quote Retrieval and Structured QA

This project implements a **Retrieval Augmented Generation (RAG)** system to semantically retrieve quotes based on natural language queries and answer questions with context-aware responses. It involves fine-tuning, embedding generation, retrieval using FAISS, and answering via a lightweight LLM.

## 📌 Objective

Build a semantic quote retrieval system that:
- Learns sentence embeddings from quotes (fine-tuned).
- Retrieves relevant quotes for user queries.
- Generates structured answers using a Retrieval Augmented Generation approach.
- Offers a user-friendly interface using Streamlit.


## 🛠️ Technologies Used

| Component               | Tool/Library Used                        
|------------------------|-------------------------------------------------------------------------------------|
| Dataset                | [`Abirate/english_quotes`](https://huggingface.co/datasets/Abirate/english_quotes)  |
| Embedding Model        | `all-MiniLM-L6-v2` from SentenceTransformers                                        |
| Retriever              | `FAISS` for vector similarity search                                                |
| LLM for Generation     | `distilgpt2` (lightweight transformer for text generation)                          |
| Evaluation Framework   | `RAGAS`                                                                             |
| Frontend               | `Streamlit`                                                                         |

## 1. 🧹 Data Preparation

- Loaded the quotes from the Hugging Face dataset `Abirate/english_quotes`.
- Cleaned data: removed null values, standardized text (lowercased, trimmed), filtered invalid entries.
- Concatenated relevant fields (quote + author + tags) for meaningful embedding.

## 2. 🧠 Model Fine-Tuning

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Task: Retrieve relevant quotes based on a user query (e.g., "Quotes about happiness by Oscar Wilde").
- Fine-tuned using cosine similarity loss on quote-query pairs.
- Saved model locally (`/model/` directory).

## 3. 🔁 RAG Pipeline

### ✅ Retrieval:
- Quotes were embedded using the fine-tuned model.
- FAISS index built for fast approximate nearest neighbor search.
- On query input, the system retrieves top-k similar quotes based on vector similarity.

### 💬 Generation:
- Retrieved quote context passed to `distilgpt2`.
- Prompt engineering used to construct coherent answers from retrieved data.
- Structured output includes:
  - Relevant quotes
  - Author(s)
  - Tags
  - Optional: similarity score

## 4. 📊 RAG Evaluation

- Framework Used: **RAGAS**
- Sample query set evaluated for:
  - **Context Relevance**
  - **Answer Quality**
  - **Retrieval Accuracy**

### 📈 Example Output:
```
context_relevance: 0.202
answer_quality: 0.150
retrieval_accuracy: 1.000
```

> **Note:** While retrieval is accurate, there is room for improvement in contextual answer quality—likely due to the lightweight nature of `distilgpt2`.

## 5. 🌐 Streamlit App

### Features:
- Input: Natural language queries (e.g., “Show me quotes about love by women authors”)
- Output: JSON response with:
  - Quotes
  - Authors
  - Tags
  - Summary (generated)
  - Optional: similarity scores
- Responsive UI with clean formatting and optional explanation modal.

To run locally:

```bash
streamlit run app.py
```

## 📦 Installation & Setup

```bash
# Clone the repository
git clone <repo-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

## 🧪 Evaluation & Limitations

### Pros:
- Accurate quote retrieval (100% retrieval accuracy for test set).
- Lightweight and fast inference using `distilgpt2`.
- Interactive and clean UI with structured responses.

### Limitations:
- `distilgpt2` is not highly capable for nuanced summarization or deep semantic generation.
- Evaluation metrics (context relevance & answer quality) were relatively low—can be improved using a stronger LLM like GPT-4 or LLaMA-3.
- Dataset is small and generic; may not cover niche or complex queries well.

## 🎥 Demo Video

A short screen recording is included (`demo_video.mp4`) explaining:
- Code overview
- Model training
- RAG pipeline
- Streamlit interface in action


## 📌 Future Improvements

- Replace `distilgpt2` with more powerful open-weight models like Mistral-7B or LLaMA-3-Instruct.
- Add metadata filtering (e.g., author gender, quote length).
- Improve evaluation with human feedback loop or Quotient.
