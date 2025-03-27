import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

# Load your JSON file (ensure correct path)
with open("rag_chunks.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract text content from JSON
texts = [entry["text"] for entry in data]

# Load a high-accuracy embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model.to("cpu")

# Generate embeddings for the text data
embeddings = embedding_model.encode(texts, convert_to_numpy=True).tolist()

# Connect to local Qdrant instance
qdrant = QdrantClient(path="qdrant_storage")

# Define the collection name
collection_name = "rag_hybrid_search"

# Create the collection with Cosine distance for high accuracy
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
)

# Prepare data points for Qdrant indexing
points = [
    PointStruct(id=i, vector=embeddings[i], payload={"text": texts[i]})
    for i in range(len(texts))
]

# Upload the indexed data to Qdrant
qdrant.upsert(collection_name=collection_name, points=points)

print(f"Successfully indexed {len(texts)} documents in Qdrant!")
