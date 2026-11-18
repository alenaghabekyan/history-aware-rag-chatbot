from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import os


model = ChatOllama(
    model="llama3.2",
    temperature=0,
    base_url=os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )
)
def format_history(chat_history):
    formatted_history = []


    for message in chat_history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            formatted_history.append(f"User:{content}")
        elif role == "assistant":
            formatted_history.append(f"Assistant:{content}")
        elif role == "system":
            formatted_history.append(f"Previous conversation summary: {content}")

    return "\n".join(formatted_history)


def rewrite_query(question, chat_history):
    if not chat_history:
        return question


    history = format_history(chat_history)
    messages = [
        SystemMessage(
            content="""
            You are a query rewriting assistant.

            Your task is to rewrite the user's latest question
            as a standalone search query using conversation history.

            Rules:
            - Resolve references such as it, they, this, that.
            - Preserve the original meaning.
            - Do not answer the question.
            - Return only the rewritten question.
            """
        ),
        HumanMessage(
            content=f"""
            Conversation history:
            {history}

            Latest user question:
            {question}

            Rewritten standalone search query:
            """
        )
    ]

    result = model.invoke(messages)

    rewritten_query = result.content.strip()

    return rewritten_query if rewritten_query else question


def generate_history_aware_answer(question, relevant_docs,chat_history):
    if not relevant_docs:
        return "I don't have enough information to answer this question."
    history = format_history(chat_history)

    context = "\n\n".join(
        f"Document {i}:\n{doc.page_content}"
        for i, doc in enumerate(relevant_docs, 1)
    )

    messages = [
        SystemMessage(
            content="""
            You are a precise question-answering assistant.

            Use the retrieved documents as your factual source.
            Use conversation history to understand the user's question.
            Do not invent facts.
            If the documents do not contain the answer,
            say that there is not enough information.
            """
        ),
        HumanMessage(
            content=f"""
            Conversation history:
            {history}

            Retrieved documents:
            {context}

            Current user question:
            {question}

            Answer the current question clearly and concisely.
            """
        )
    ]

    result = model.invoke(messages)

    return result.content.strip()
