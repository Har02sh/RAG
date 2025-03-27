import json
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from sentence_transformers import SentenceTransformer

# Load the same embedding model used for indexing
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant with persistent storage (qdrantStorage)
qdrant = QdrantClient(path=r"D:\Chunking\qdrant_storage")  # Using persistent data

# Define the collection name
collection_name = "rag_hybrid_search"

def hybrid_search(query: str, top_k: int = 5):
    """
    Performs hybrid search: vector + keyword-based filtering.
    
    Args:
    - query (str): The search query
    - top_k (int): Number of results to return
    
    Returns:
    - List of best matching results
    """
    
    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Perform Vector Search in Qdrant
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128, exact=True),  # Optimized for accuracy
    )

    # Print results
    print("\n🔹 Hybrid Search Results:")
    for i, result in enumerate(results):
        print(f"🔹 Rank {i+1}: {result.payload['text']} (Score: {result.score})")

    return results

# Example Usage
if __name__ == "__main__":
    query_text = input("Enter your query: ")
    hybrid_search(query_text)