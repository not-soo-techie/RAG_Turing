import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader


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
        print(f"Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content: {doc.page_content[:100]}...")
        
    return documents

def main():
    print("Starting ingestion process")
    
    # Load documents
    
    # Split document into chunks
    
    # Create Embeddings
    
    # Store embeddings in vector db
    

if __name__ == "__main__":
    main()