



from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import os

def generate_answer(query, relevant_docs):

    if not relevant_docs:
        return "I don't have enough information to answer this question."

    context = "\n\n".join(
        f"Document {i}:\n{doc.page_content}"
        for i, doc in enumerate(relevant_docs, 1)
    )

    model = ChatOllama(
        model="llama3.2",
        temperature=0,
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )
    )
    messages = [
        SystemMessage(
            content="""
            You are a factual question-answering assistant.
            Answer using the provided context only.
            If the context contains the answer, you must provide it.
            Do not say information is missing when it is explicitly present.
            """
        ),
        HumanMessage(
            content=f"""
            Context:
            {context}

            Question:
            {query}

            Give a short, direct answer.
            """
        )
    ]

    result = model.invoke(messages)

    return result.content
