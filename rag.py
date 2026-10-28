
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Locate the persistent Chroma database beside this file
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# Use the same embedding model used to build the index
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to the existing Chroma collection
db = Chroma(
    collection_name="pdf_knowledge",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

# Connect to the local Ollama model
# Connect to Gemini Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyDC7rlshtkQY_1yWXinUAKJZdcq43QmWnI",
    temperature=0,
)



def ask_question(question: str) -> dict:
    """Retrieve PDF context and generate an answer."""

    # 1. Retrieve the three most relevant chunks
    retrieved_docs = db.similarity_search(question, k=3)

    if not retrieved_docs:
        return {
            "answer": "I couldn't find relevant information in the indexed PDFs.",
            "sources": [],
        }

    # 2. Prepare context and source references
    context_parts = []
    sources = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"Source: {source}, Page: {page}\n{doc.page_content}"
        )

        reference = {"source": source, "page": page}
        if reference not in sources:
            sources.append(reference)

    context = "\n\n---\n\n".join(context_parts)

    # 3. Build the prompt sent to the LLM
    prompt = f"""
You are a helpful assistant answering questions about PDF documents.

Answer the question using only the supplied context.
If the context does not contain the answer, say that you
could not find sufficient information in the documents.
Do not invent facts or citations.

Context:
{context}

Question:
{question}

Answer:
"""

    # 4. Ask Ollama to generate the answer
    response = llm.invoke(prompt)

    # 5. Return the answer and its source references
    return {
        "answer": response.content,
        "sources": sources,
    }


if __name__ == "__main__":
    question = input("Ask a question about your PDFs: ").strip()

    if question:
        result = ask_question(question)

        print("\n=== Answer ===")
        print(result["answer"])

        print("\n=== Sources retrieved ===")
        for source in result["sources"]:
            print(f"- {source['source']} | Page {source['page']}")
