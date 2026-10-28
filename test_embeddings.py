
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Two sentences with related meanings
sentences = [
    "RAG retrieves information from documents to answer questions.",
    "An AI system searches a knowledge base before generating a response.",
    "I enjoy eating pizza on weekends.",
]

# Convert each sentence into a numerical vector
embeddings = model.encode(sentences)

# Inspect the results
print("Number of sentences:", len(sentences))
print("Embedding dimensions:", embeddings.shape)
print("First 10 numbers of the first vector:")
print(embeddings[0][:10])
