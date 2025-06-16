# RAG-Based Semantic Quote Retrieval and Structured QA

This project implements a **Retrieval Augmented Generation (RAG)** system to semantically retrieve quotes based on natural language queries and answer questions with context-aware responses. It involves fine-tuning, embedding generation, retrieval using FAISS, and answering via a lightweight LLM.

## Objective

Build a semantic quote retrieval system that:
- Learns sentence embeddings from quotes (fine-tuned).
- Retrieves relevant quotes for user queries.
- Generates structured answers using a Retrieval Augmented Generation approach.
- Offers a user-friendly interface using Streamlit.


## Technologies Used

| Component               | Tool/Library Used                        
|------------------------|-------------------------------------------------------------------------------------|
| Dataset                | [`Abirate/english_quotes`](https://huggingface.co/datasets/Abirate/english_quotes)  |
| Embedding Model        | `DistilBERT` from SentenceTransformers                                        |
| Retriever              | `FAISS` for vector similarity search                                                |
| LLM for Generation     | `Ollama`                         |
| Evaluation Framework   | `RAGAS`                                                                             |
| Frontend               | `Streamlit`                                                                         |

## 1. Data Preparation

- Loaded the quotes from the Hugging Face dataset `Abirate/english_quotes`.
- Cleaned data: removed null values, standardized text (lowercased, trimmed), filtered invalid entries.
- Concatenated relevant fields (quote + author + tags) for meaningful embedding.

## 2. Model Fine-Tuning

- Model: `DistilBERT`
- Task: Retrieve relevant quotes based on a user query (e.g., "Quotes about happiness by Oscar Wilde").
- Fine-tuned using cosine similarity loss on quote-query pairs.
- Saved model locally (`/model/` directory).

## 3. RAG Pipeline

### Retrieval:
- Quotes were embedded using the fine-tuned model.
- FAISS index built for fast approximate nearest neighbor search.
- On query input, the system retrieves top-k similar quotes based on vector similarity.

### Generation:
- Retrieved quote context passed to `Ollama`.
- Prompt engineering used to construct coherent answers from retrieved data.
- Structured output includes:
  - Relevant quotes
  - Author(s)
  - Tags
  - Optional: similarity score

## 4.RAG Evaluation

- Framework Used: **RAGAS**
- Sample query set evaluated for:
  - **Context Relevance**
  - **Answer Quality**
  - **Retrieval Accuracy**


## 5. Streamlit App

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

## Installation & Setup

```bash
# Clone the repository
git clone <repo-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```
