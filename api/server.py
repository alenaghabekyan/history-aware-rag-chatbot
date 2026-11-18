
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.retrival_pipeline import retrieve_documents

from src.history_aware_generation import (
    rewrite_query,
    generate_history_aware_answer
)

from src.chat_history import (
    init_db,
    get_history,
    save_message
)

from src.source_citations import format_sources


# =====================================
# FastAPI Application
# =====================================

app = FastAPI(
    title="RAG Chatbot API",
    description="A History-Aware RAG Application",
    version="1.0.0"
)


# =====================================
# Request and Response Models
# =====================================

class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# =====================================
# Startup
# =====================================

@app.on_event("startup")
def startup_event():
    init_db()


# =====================================
# Home Endpoint
# =====================================

@app.get("/")
def home():
    return {
        "message": "RAG Chatbot API is running"
    }


# =====================================
# Health Check
# =====================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =====================================
# Chat Endpoint
# =====================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        # 1. Get previous conversation
        chat_history = get_history(request.session_id)

        # 2. Rewrite question using conversation history
        standalone_query = rewrite_query(
            request.question,
            chat_history
        )

        print("\n========== QUERY REWRITING ==========")
        print("Original question:", request.question)
        print("Rewritten question:", standalone_query)

        # 3. Retrieve documents using rewritten query
        relevant_docs = retrieve_documents(standalone_query)

        print("\n========== RETRIEVAL ==========")
        print("Retrieved documents:", len(relevant_docs))

        # 4. Generate answer using original question,
        # retrieved documents and conversation history
        answer = generate_history_aware_answer(
            request.question,
            relevant_docs,
            chat_history
        )

        print("\n========== GENERATED ANSWER ==========")
        print(answer)

        # 5. Format source citations
        sources = format_sources(relevant_docs)

        # 6. Save user question
        save_message(
            request.session_id,
            "user",
            request.question
        )

        # 7. Save assistant answer
        save_message(
            request.session_id,
            "assistant",
            answer
        )

        # 8. Return API response
        return ChatResponse(
            answer=answer,
            sources=sources
        )

    except Exception as e:
        print("ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
