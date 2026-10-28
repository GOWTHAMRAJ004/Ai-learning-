
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Project directories
PDF_DIR = Path("documents")

def load_pdfs():
    """Extract text from every PDF and preserve page metadata."""
    documents = []

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found. Add PDFs to the documents folder."
        )

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        with fitz.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_number,
                            },
                        )
                    )

    return documents


def split_documents(documents):
    """Split each page's text into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    pages = load_pdfs()
    chunks = split_documents(pages)

    print(f"\nPDF pages with text: {len(pages)}")
    print(f"Text chunks created: {len(chunks)}")

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0].page_content[:500])
        print("\nMetadata:", chunks[0].metadata)
