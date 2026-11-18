
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


def summarize_conversation(old_summary, old_messages):

    conversation = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in old_messages
    )

    messages = [
        SystemMessage(content="""
        You are a conversation summarization assistant.

        Create a concise summary of the conversation.
        Preserve:
        - Important facts
        - User's goals and preferences
        - Topics already discussed
        - Important details needed for future questions

        Do not invent information.
        Return only the summary.
        """),

        HumanMessage(content=f"""
        Previous summary:
        {old_summary if old_summary else "No previous summary."}

        New conversation messages:
        {conversation}

        Updated summary:
        """)
    ]

    result = model.invoke(messages)

    return result.content.strip()
