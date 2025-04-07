import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchText, SearchParams
from sentence_transformers import SentenceTransformer

# Load the same embedding model used for indexing
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant with persistent storage
qdrant = QdrantClient("localhost", port=6333)

# Define the collection name
collection_name = "rag_hybrid_search"

def hybrid_search(query: str, top_k: int = 5):
    """
    Performs hybrid search: vector search with keyword-based filtering.
    
    Args:
    - query (str): The search query
    - top_k (int): Number of results to return
    
    Returns:
    - List of best matching results
    """
    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()
    
    # Create a text match condition for keyword search
    keyword_filter = Filter(
        must=[
            FieldCondition(
                key="text",
                match=MatchText(text=query)
            )
        ]
    )
    
    # Perform vector search with keyword filtering in Qdrant
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        query_filter=keyword_filter,  # Apply text-based filtering
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128, exact=True),  # Optimized for accuracy
        with_payload=True,
        with_vectors=False  # Set to True if you need the vectors in results
    )

    # Print results
    print("\n🔹 Hybrid Search Results:")
    for i, result in enumerate(results):
        print(f"🔹 Rank {i+1}: {result.payload.get('text', 'No text available')} (Score: {result.score:.4f})")
        # Optionally print additional payload fields if available
        for key, value in result.payload.items():
            if key != 'text':
                print(f"  • {key}: {value}")

    return results

def two_stage_hybrid_search(query: str, top_k: int = 5, vector_weight: float = 0.7, keyword_weight: float = 0.3):
    """
    Implements a manual two-stage hybrid search approach.
    First performs vector search, then reranks results using text similarity.
    
    Args:
    - query (str): The search query
    - top_k (int): Number of results to return
    - vector_weight (float): Weight for vector similarity (0.0-1.0)
    - keyword_weight (float): Weight for keyword similarity (0.0-1.0)
    
    Returns:
    - List of reranked results
    """
    # Get more results than needed for reranking
    search_limit = min(top_k * 3, 100)  # Get 3x results or up to 100
    
    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()
    
    # First stage: Vector search
    vector_results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=search_limit,
        search_params=SearchParams(hnsw_ef=128, exact=True),
        with_payload=True,
        with_vectors=False
    )
    
    if not vector_results:
        return []
    
    # Normalize vector scores (min-max normalization)
    max_vector_score = max(result.score for result in vector_results)
    min_vector_score = min(result.score for result in vector_results)
    score_range = max_vector_score - min_vector_score
    
    # Second stage: Calculate text similarity and combine scores
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Extract text from results
    documents = [result.payload.get('text', '') for result in vector_results]
    if not all(documents):
        print("Warning: Some documents don't have 'text' field in payload")
        documents = [doc if doc else "missing_text" for doc in documents]
    
    # Calculate text similarity using TF-IDF
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(documents + [query])
        document_vectors = tfidf_matrix[:-1]
        query_vector = tfidf_matrix[-1]
        
        # Calculate cosine similarities
        keyword_scores = cosine_similarity(document_vectors, query_vector).flatten()
    except:
        # Fallback if TF-IDF fails
        print("Warning: TF-IDF calculation failed, using simple text matching")
        keyword_scores = np.array([
            len(set(query.lower().split()) & set(doc.lower().split())) / max(len(query.split()), 1) 
            for doc in documents
        ])
    
    # Combine scores
    combined_results = []
    for i, result in enumerate(vector_results):
        # Normalize vector score
        if score_range > 0:
            norm_vector_score = (result.score - min_vector_score) / score_range
        else:
            norm_vector_score = 1.0
            
        # Combined score (weighted sum)
        combined_score = (vector_weight * norm_vector_score) + (keyword_weight * keyword_scores[i])
        
        combined_results.append({
            "payload": result.payload,
            "vector_score": result.score,
            "keyword_score": float(keyword_scores[i]),
            "combined_score": combined_score
        })
    
    # Sort by combined score
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Take top_k results
    top_results = combined_results[:top_k]
    
    # Print results
    print("\n🔹 Two-Stage Hybrid Search Results:")
    for i, result in enumerate(top_results):
        print(f"🔹 Rank {i+1}: {result['payload'].get('text', 'No text available')}")
        print(f"  • Vector Score: {result['vector_score']:.4f}")
        print(f"  • Keyword Score: {result['keyword_score']:.4f}")
        print(f"  • Combined Score: {result['combined_score']:.4f}")
    
    return top_results

def advanced_search_with_filters(
    query: str, 
    top_k: int = 5,
    filter_conditions: dict = None
):
    """
    Performs vector search with advanced filtering options.
    
    Args:
    - query (str): The search query
    - top_k (int): Number of results to return
    - filter_conditions (dict): Additional filtering conditions
    
    Returns:
    - List of search results
    """
    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()
    
    # Build filter conditions
    must_conditions = []
    
    # Add any additional filters
    if filter_conditions:
        for field, value in filter_conditions.items():
            if isinstance(value, list):
                # For list values, create an OR condition
                should_conditions = []
                for v in value:
                    should_conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchText(text=str(v))
                        )
                    )
                must_conditions.append(Filter(should=should_conditions))
            else:
                # For single values
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchText(text=str(value))
                    )
                )
    
    # Create the final filter (if any conditions exist)
    search_filter = Filter(must=must_conditions) if must_conditions else None
    
    # Perform search in Qdrant
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        query_filter=search_filter,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128, exact=True),
        with_payload=True,
        with_vectors=False
    )

    # Print results
    print("\n🔹 Advanced Search Results:")
    for i, result in enumerate(results):
        print(f"🔹 Rank {i+1}: {result.payload.get('text', 'No text available')} (Score: {result.score:.4f})")
        # Print additional payload fields if available
        for key, value in result.payload.items():
            if key != 'text':
                print(f"  • {key}: {value}")

    return results


# Example Usage
if __name__ == "__main__":
    print("Select search mode:")
    print("1. Basic Hybrid Search")
    print("2. Two-Stage Hybrid Search")
    print("3. Advanced Search with Filters")
    
    choice = input("Enter choice (1-3): ")
    query_text = input("Enter your query: ")
    
    if choice == "2":
        two_stage_hybrid_search(query_text)
    elif choice == "3":
        # Example of advanced search with filters
        category = input("Filter by category (optional, press Enter to skip): ")
        date_range = input("Filter by date range (optional, press Enter to skip): ")
        
        filters = {}
        if category:
            filters["category"] = category
        if date_range:
            filters["date"] = date_range
            
        advanced_search_with_filters(query_text, filter_conditions=filters)
    else:
        hybrid_search(query_text)