import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional

# PDF processing
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize

# Vector search
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchText, SearchParams
from sentence_transformers import SentenceTransformer

# For text processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# LLM integration
import openai  # For OpenAI models
import requests  # For other API-based models
from dotenv import load_dotenv

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables
load_dotenv()

# Initialize models and clients
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient("localhost", port=6333)
collection_name = "rag_hybrid_search"

class DocumentProcessor:
    """Process and index PDF documents for RAG system"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vectorizer = TfidfVectorizer(
            max_features=100, 
            stop_words='english'
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF and chunk it into sections
        Returns list of chunks with metadata
        """
        doc = fitz.open(pdf_path)
        chunks = []
        
        # Extract document-level metadata
        metadata = {
            "filename": os.path.basename(pdf_path),
            "total_pages": len(doc),
            "document_id": os.path.splitext(os.path.basename(pdf_path))[0]
        }
        
        # Process each page
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # Skip empty pages
            if not text.strip():
                continue
                
            # Extract page-level metadata (e.g., headers/footers)
            page_metadata = {
                "page_num": page_num + 1,
                **metadata
            }
            
            # Chunk the page (here using simple sentence splitting)
            sentences = sent_tokenize(text)
            
            # Group sentences into chunks (adjust chunk size as needed)
            chunk_size = 5  # sentences per chunk
            for i in range(0, len(sentences), chunk_size):
                chunk_text = " ".join(sentences[i:i+chunk_size])
                
                # Skip very small chunks
                if len(chunk_text.split()) < 10:
                    continue
                
                # Extract chunk-level metadata
                chunk_metadata = {
                    "chunk_id": f"{page_num+1}-{i//chunk_size+1}",
                    **page_metadata
                }
                
                # Extract and add keywords
                keywords = self._extract_keywords(chunk_text)
                chunk_metadata["keywords"] = keywords
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
        
        return chunks
    
    def _extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords from text using TF-IDF"""
        # For a single document, use CountVectorizer instead
        cv = CountVectorizer(max_features=num_keywords, stop_words='english')
        
        try:
            counts = cv.fit_transform([text])
            keywords = cv.get_feature_names_out()
            return keywords.tolist()
        except:
            # Fallback to simple word frequency
            words = re.findall(r'\b\w{3,}\b', text.lower())
            word_counts = {}
            for word in words:
                if word not in ['the', 'and', 'for', 'with', 'that']:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_words[:num_keywords]]
    
    def index_document(self, pdf_path: str) -> List[str]:
        """
        Process PDF document and index it in Qdrant
        Returns list of chunk IDs
        """
        # Extract and chunk text
        chunks = self.extract_text_from_pdf(pdf_path)
        
        # Prepare vectors and payloads for batch upsert
        ids = []
        vectors = []
        payloads = []
        
        for i, chunk in enumerate(chunks):
            # Create a unique ID
            chunk_id = f"{chunk['metadata']['document_id']}_{chunk['metadata']['chunk_id']}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(chunk['text'])
            
            # Prepare payload with text and all metadata
            payload = {
                "text": chunk['text'],
                **chunk['metadata']
            }
            
            ids.append(chunk_id)
            vectors.append(embedding)
            payloads.append(payload)
        
        # Batch upsert to Qdrant
        qdrant.upsert(
            collection_name=collection_name,
            points=list(zip(ids, vectors, payloads))
        )
        
        return ids


class QueryAnalyzer:
    """Analyze user queries to extract search parameters"""
    
    def __init__(self):
        # Common temporal phrases
        self.temporal_patterns = [
            (r'in\s+(\d{4})', 'year'),
            (r'page\s+(\d+)', 'page_num'),
            (r'chapter\s+(\d+|[ivxlcdm]+)', 'chapter'),
            (r'section\s+(\d+\.\d+|\d+)', 'section'),
            (r'part\s+(\d+|[ivxlcdm]+)', 'part')
        ]
    
    def extract_search_parameters(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze query to extract search constraints
        Returns cleaned query and filter parameters
        """
        # Initialize parameters
        filters = {}
        cleaned_query = query
        
        # Check for metadata parameters in the query
        for pattern, field in self.temporal_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                filters[field] = matches[0]
                # Remove the pattern from the query
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
        
        # Clean up query (remove extra spaces, etc.)
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        
        return cleaned_query, filters


class LLMInterface:
    """Interface to language models for answer generation"""
    
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        
        # Setup based on model type
        if model_type == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Add more model types as needed:
        # - Anthropic Claude
        # - Hugging Face models
        # - Local models like LLaMA, etc.
    
    def generate_answer(self, question: str, context: str, document_info: Dict[str, Any] = None) -> str:
        """
        Generate an answer using an LLM
        """
        if self.model_type == "openai":
            return self._generate_with_openai(question, context, document_info)
        # Add other model implementations as needed
        
        # Default fallback
        return "Sorry, I couldn't generate an answer with the configured model."
    
    def _generate_with_openai(self, question: str, context: str, document_info: Dict[str, Any] = None) -> str:
        """Generate answer using OpenAI API"""
        
        # Create system prompt with document information if available
        system_prompt = "You are a helpful assistant that answers questions based on the provided context."
        
        if document_info:
            doc_name = document_info.get("filename", "the document")
            system_prompt += f" The context is from {doc_name}."
            
            # Add more document details if available
            if "total_pages" in document_info:
                system_prompt += f" The document has {document_info['total_pages']} pages."
        
        # Create messages for the chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer the question based only on the provided context. If the context doesn't contain enough information to answer the question, state that the answer cannot be determined from the given context."}
        ]
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=1000
            )
            
            # Extract answer from response
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"


class DocumentQA:
    """Main class for document QA system"""
    
    def __init__(self, llm_type="openai"):
        self.embedding_model = embedding_model
        self.qdrant = qdrant
        self.collection_name = collection_name
        self.document_processor = DocumentProcessor(embedding_model)
        self.query_analyzer = QueryAnalyzer()
        self.llm = LLMInterface(model_type=llm_type)
    
    def index_document(self, pdf_path: str) -> List[str]:
        """Index a document in the vector database"""
        return self.document_processor.index_document(pdf_path)
    
    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a question about indexed documents"""
        # Analyze the query to extract search parameters
        cleaned_query, query_filters = self.query_analyzer.extract_search_parameters(question)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(cleaned_query).tolist()
        
        # Build filter conditions
        must_conditions = []
        
        # Add filters extracted from query
        if query_filters:
            for field, value in query_filters.items():
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchText(text=str(value))
                    )
                )
        
        # Create the final filter (if any conditions exist)
        search_filter = Filter(must=must_conditions) if must_conditions else None
        
        # Perform vector search
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=128, exact=True),
            with_payload=True,
            with_vectors=False
        )
        
        # Extract relevant text for context
        context_texts = []
        document_info = {}
        
        for i, result in enumerate(results):
            # Extract text
            text = result.payload.get('text', '')
            score = result.score
            
            # Format with metadata for context
            metadata_str = f"[Document: {result.payload.get('document_id', 'Unknown')}, "
            metadata_str += f"Page: {result.payload.get('page_num', 'Unknown')}, "
            metadata_str += f"Relevance: {score:.2f}]"
            
            context_texts.append(f"{metadata_str}\n{text}")
            
            # Gather document info from first result
            if i == 0 and not document_info:
                document_info = {
                    "filename": result.payload.get('filename', 'Unknown'),
                    "total_pages": result.payload.get('total_pages', 0),
                    "document_id": result.payload.get('document_id', 'Unknown')
                }
        
        # Combine context
        context = "\n\n".join(context_texts)
        
        # Generate answer
        answer = self.llm.generate_answer(question, context, document_info)
        
        # Return complete result
        return {
            "query": question,
            "cleaned_query": cleaned_query,
            "extracted_filters": query_filters,
            "context": context,
            "answer": answer,
            "results": results
        }


# Example usage
if __name__ == "__main__":
    # Use 'openai' by default, but could be changed to other models
    llm_type = os.getenv("LLM_TYPE", "openai")
    doc_qa = DocumentQA(llm_type=llm_type)
    
    # Check if necessary keys are set
    if llm_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Set this in your .env file.")
    
    # Ask if user wants to index a document
    index_choice = input("Do you want to index a PDF document? (y/n): ")
    
    if index_choice.lower() == 'y':
        pdf_path = input("Enter path to PDF file: ")
        try:
            chunks = doc_qa.index_document(pdf_path)
            print(f"Successfully processed and indexed document")
        except Exception as e:
            print(f"Error indexing document: {str(e)}")
    
    # Process questions
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        try:
            # Get answer
            result = doc_qa.answer_question(question)
            
            # Display answer
            print("\n" + "="*80)
            print("📝 ANSWER:")
            print(result["answer"])
            print("="*80)
            
            # Display extracted filters if any
            if result["extracted_filters"]:
                print("\n📋 Automatically extracted filters:")
                for field, value in result["extracted_filters"].items():
                    print(f"  • {field}: {value}")
            
            # Ask if user wants to see supporting evidence
            show_evidence = input("\nShow supporting evidence? (y/n): ")
            if show_evidence.lower() == 'y':
                print("\n🔍 Supporting Evidence:")
                for i, chunk in enumerate(result["results"][:3]):  # Show top 3 for brevity
                    print(f"\n📄 Source {i+1} (Score: {chunk.score:.4f})")
                    print(f"  • Document: {chunk.payload.get('document_id', 'Unknown')}")
                    print(f"  • Page: {chunk.payload.get('page_num', 'Unknown')}")
                    
                    # Display text snippet
                    text = chunk.payload.get('text', 'No text available')
                    print(f"\n{text[:300]}..." if len(text) > 300 else f"\n{text}")
        
        except Exception as e:
            print(f"Error processing question: {str(e)}")