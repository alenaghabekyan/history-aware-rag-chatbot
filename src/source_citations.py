import os

def format_sources(relevant_docs):
    sources = []
    seen_sources = set()

    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown source")
        source = os.path.basename(source)

        if source not in seen_sources:
            seen_sources.add(source)
            sources.append({"source": source})

    return sources


def print_sources(sources):
    print("\n--- SOURCES ---")

    if not sources:
        print("No sources found.")
        return

    for i, source in enumerate(sources, 1):
        print(f"{i}. {source['source']}")

    print("---------------")