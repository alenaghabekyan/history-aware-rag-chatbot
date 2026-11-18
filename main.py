from src.retrival_pipeline import retrieve_documents
from src.history_aware_generation import rewrite_query, generate_history_aware_answer
from src.coversation_summarization import summarize_conversation
from src.chat_history import (
    init_db,
    save_message,
    get_recent_history,
    get_history,
    get_summary,
    save_summary,
    keep_recent_messages,
    clear_history
)
from src.source_citations import format_sources, print_sources

def main():

    # Initialize SQLite database
    init_db()


    # For now, we use one persistent conversation
    session_id = "default"
    summary = get_summary(session_id)
    chat_history = get_recent_history(session_id, limit=6)

    if summary:
        chat_history.insert(0, {
        "role": "system",
        "content": summary
    })

    # Load previous conversation
    chat_history = get_recent_history(session_id)

    print("History-Aware RAG is running!")
    print("Previous messages:", len(chat_history))
    print("Type 'exit' to stop.\n")
    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
         break

        if question.lower() == "clear":
            clear_history(session_id)
            chat_history = []
            print("Chat history cleared!")
            continue

        if not question:
            continue

        # 1. Rewrite query using history
        search_query = rewrite_query(question, chat_history)

        print("\nRewritten Query:", search_query)

        # 2. Retrieve documents
        relevant_docs = retrieve_documents(search_query)
        print("\n--- RETRIEVED DOCUMENT METADATA ---")

        for i, doc in enumerate(relevant_docs, 1):
            print(f"\nDocument {i}")
            print("Metadata:", doc.metadata)
            print("Content:", doc.page_content[:200])

        print("----------------------------------\n")

        # 3. Generate answer
        answer = generate_history_aware_answer(
            question,
            relevant_docs,
            chat_history
        )

        print("\nAI:", answer)
        
        sources = format_sources(relevant_docs)
        print_sources(sources)

        # 4. Save messages
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)

        # 5. Check whether summarization is needed
        all_messages = get_history(session_id)

        if len(all_messages) > 10:

            old_messages = all_messages[:-6]

            new_summary = summarize_conversation(
                get_summary(session_id),
                old_messages
            )

            save_summary(session_id, new_summary)

            keep_recent_messages(session_id, limit=6)

            print("\n--- CONVERSATION SUMMARY UPDATED ---")
            print(new_summary)

        # 6. Reload summary and recent history
        summary = get_summary(session_id)
        chat_history = get_recent_history(session_id, limit=6)

        if summary:
            chat_history.insert(0, {
                "role": "system",
                "content": summary
            })


if __name__ == "__main__":
    main()
