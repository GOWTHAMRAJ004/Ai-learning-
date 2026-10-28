
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import load_pdfs, split_documents

# Store the database inside this project
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# Load the local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def main():
    # Step 1: Extract and split the PDFs
    pages = load_pdfs()
    chunks = split_documents(pages)

    print(f"Pages loaded: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    # Step 2: Create a persistent Chroma database
    db = Chroma(
        collection_name="pdf_knowledge",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Step 3: Add the chunks to Chroma
    db.add_documents(chunks)

    print(f"Chunks stored: {len(chunks)}")
    print(f"Database location: {CHROMA_DIR}")

    # Step 4: Test semantic retrieval
    question = "What is retrieval augmented generation?"
    results = db.similarity_search(question, k=3)

    print(f"\nTop results for: {question}")

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:400])
        print("Source:", doc.metadata)

if __name__ == "__main__":
    main()
