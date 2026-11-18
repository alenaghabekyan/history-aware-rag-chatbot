# History-Aware RAG Chatbot

A Retrieval-Augmented Generation (RAG) application built with Python, LangChain, ChromaDB, Ollama, FastAPI, and Docker.

The system retrieves relevant documents, understands conversation context, generates answers using a local LLM, and provides source citations.

## Features

- Document ingestion and text splitting
- Semantic search with vector embeddings
- ChromaDB vector database
- History-aware query rewriting
- Persistent conversation history with SQLite
- Conversation summarization
- Source citations
- Retrieval evaluation
- REST API with FastAPI
- Docker containerization
- Local LLM inference using Ollama

## Architecture

```text
User Question
     |
     v
FastAPI
     |
     v
Chat History (SQLite)
     |
     v
History-Aware Query Rewriting
     |
     v
ChromaDB Retriever
     |
     v
Relevant Documents
     |
     v
Ollama (Llama 3.2)
     |
     v
Generated Answer
     |
     v
Source Citations
     |
     v
API Response
```

## Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Ollama (Llama 3.2)
- FastAPI
- SQLite
- Docker

## Embedding Model

`sentence-transformers/all-MiniLM-L6-v2`

## Language Model

`llama3.2`

## Project Structure

```text
Rag-for-begginer/
├── api/
│   └── server.py
├── docs/
├── docs_clean/
├── src/
│   ├── ingestion_pipeline.py
│   ├── retrival_pipeline.py
│   ├── answer_generation.py
│   ├── history_aware_generation.py
│   ├── chat_history.py
│   ├── coversation_summarization.py
│   ├── source_citations.py
│   ├── evaluation.py
│   └── evaluation_dataset.py
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Rag-for-begginer
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install and start Ollama, then download the model:

```bash
ollama pull llama3.2
```

## Run Locally

```bash
uvicorn api.server:app --reload
```

Open the API documentation:

`http://localhost:8000/docs`

## Run with Docker

Build the image:

```bash
docker build -t rag-chatbot .
```

Run the container:

```bash
docker run -d \
  --name rag-chatbot \
  -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v "$(pwd)/db/chroma_db:/app/db/chroma_db" \
  -v "$(pwd)/chat_history.db:/app/chat_history.db" \
  rag-chatbot
```

Make sure Ollama is running on the host machine and the `llama3.2` model is available.

## API Usage

Endpoint:

`POST /chat`

Example request:

```json
{
  "session_id": "user-123",
  "question": "Who founded Tesla?"
}
```

Example response:

```json
{
  "answer": "Tesla was founded by Martin Eberhard and Marc Tarpenning in July 2003.",
  "sources": [
    {
      "source": "tesla.txt"
    }
  ]
}
```

## Retrieval Evaluation

The retrieval evaluation dataset contains 5 questions.

| Metric | Result |
|---|---:|
| Recall@5 | 100% |
| MRR | 0.7000 |
| Precision@5 | 36% |
| NDCG@5 | 0.7565 |

These results are based on keyword-based relevance checks and a small evaluation dataset. They are preliminary, not a human-judged benchmark.

## Future Improvements

- Expand the evaluation dataset
- Improve retrieval quality
- Add automated tests
- Add authentication and API rate limiting
- Deploy the application to a cloud platform

## License

Add a license before public distribution.
