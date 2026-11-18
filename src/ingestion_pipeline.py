import os
from langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings



load_dotenv()

def load_documents(docs_path = "docs"):
    """Load all text files from the docs directory"""
    # print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} doesnot exist. please create it and addyour company files ")


    loader = DirectoryLoader(
        path=docs_path, 
        glob="*.txt",
        loader_cls= TextLoader
                             )    
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"no .txt files found in {docs_path}. Please add your company documents")

    # for i , doc in enumerate(documents[:2]):
    #     print(f"\nDocument{i+1}:")
    #     print(f"Source:{doc.metadata['source']}")
    #     print(f" ContentLength:{len(doc.page_content)} Character")
    #     print(f"Content Preview:{doc.page_content[:100]}...")
    #     print(f"metadata:{doc.metadata}")
    return documents    

def split_documents(documents, chunk_size = 800, chunk_overlap = 40):
    """Split documents into smaller chunks with overlap"""
    print("Spliting documents into chunks ...")
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # if chunks:
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n--- Chunk {i+1} ---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-" * 50)
    #     if len(chunks) > 5:
    #         print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks    
    

def create_vectore_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vectore store"""
    print("Creating embeddings and storing in chromaDB")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vector store from {persist_directory}")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
    print("--- Creating vector store ---")


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding= embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    ) 
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")

    print("Absolute path:", os.path.abspath(persist_directory))
    return vectorstore

def main():
    print("main function")

    documents = load_documents(docs_path="docs_clean")

    chunks = split_documents(documents)
    vectorstore = create_vectore_store(chunks)
if __name__ == "__main__":
    main()


