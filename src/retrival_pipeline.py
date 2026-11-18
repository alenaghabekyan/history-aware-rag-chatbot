

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PERSISTENT_DIRECTORY = "db/chroma_db"


def create_retriever():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=PERSISTENT_DIRECTORY,
        embedding_function=embedding_model
    )

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.5
        }
    )

    return retriever


def retrieve_documents(query):
    retriver = create_retriever()
    relevant_docs = retriver.invoke(query)

    return relevant_docs


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"