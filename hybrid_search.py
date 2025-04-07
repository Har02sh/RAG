import json
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from sentence_transformers import SentenceTransformer
import ollama

# Load the same embedding model used for indexing
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant with persistent storage (qdrantStorage)
qdrant = QdrantClient(host="localhost", port=6333)

# Define the collection name
collection_name = "DB1"

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
    # print("\n🔹 Hybrid Search Results:")
    # for i, result in enumerate(results):
    #     print(f"🔹 Rank {i+1}: {result.payload['text']} (Score: {result.score})")

    return results

def generate_answer_with_mistral(question: str, context: str) -> str:
        """
        Generate an answer using the Mistral model via Ollama based on the retrieved document context.
        """
        prompt = f"""You are an expert assistant. Answer the user's question using only the provided context.

        Context:
        {context}

        Question: {question}

        Provide a well-structured, concise response based on the context.
        """

        try:
            response = ollama.chat(
                model="mistral:7b",  # Ensure the Mistral model is available in Ollama
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Example Usage
if __name__ == "__main__":
    while(True):
        query_text = input("Enter your query: ")
        if query_text.lower() == "exit":
            break
        results = hybrid_search(query_text)

        # Extract text from results to use as context
        context = "\n".join([hit.payload["text"] for hit in results if "text" in hit.payload])

        # Generate answer using retrieved context
        answer = generate_answer_with_mistral(query_text, context)

        print("\n🔹 Generated Answer:")
        print(answer)