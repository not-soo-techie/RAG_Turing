import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(doc_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {doc_path}...")

    # Check if the directory exists
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Directory {doc_path} does not exist. Please create it and add your documents.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        doc_path, 
        glob="**/*.txt", 
        loader_cls=TextLoader
    )
    
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No .txt files found in {doc_path}. Please add some documents to ingest.")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content: {doc.page_content[:100]}...")  # Print the first 200 characters of the document
        print(f" Metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f" Source: {chunk.metadata['source']}")
            print(f" Length: {len(chunk.page_content)} characters")
            print(f" Content: {chunk.page_content}...")  
            print("-" * 50)
            
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks.")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embedding and storing in ChromaDB...")
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("--- Creating vector store ---")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vector store created and saved to {persist_directory}.")
    return vectorstore

def main():
    
    print("Starting ingestion process...")

    # 1. Load documents
    # loader = DirectoryLoader("path/to/documents", glob="**/*.txt", loader_cls=TextLoader)
    # documents = loader.load()
    documents = load_documents(doc_path="docs")

    # 2. Split documents into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # chunks = text_splitter.split_documents(documents)
    chunks = split_documents(documents)

    # 3. Create embeddings
    # embeddings = OpenAIEmbeddings()
    # embedded_chunks = embeddings.embed_documents(chunks)

    # 4. Store embeddings
    # db = Chroma(embedding_function=embeddings)
    # db.add_documents(embedded_chunks)
    vectorstore = create_vector_store(chunks)

    # print("Ingestion process completed.")

if __name__ == "__main__":
    main()