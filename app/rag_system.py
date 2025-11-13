"""
RAG System for Product Reviews
Handles embedding creation, vector storage, and retrieval
"""

import os
import pickle
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Tuple, Iterator


class RAGSystem:
    def __init__(self, openai_api_key: str, data_path: str = None):
        """
        Initialize the RAG system
        
        Args:
            openai_api_key: OpenAI API key for embeddings and generation
            data_path: Path to the reviews CSV file
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        self.generation_model = "gpt-4o-mini"  # Fast and cost-effective model
        self.index = None
        self.reviews_df = None
        self.embeddings = None
        
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """Load review data from CSV"""
        self.reviews_df = pd.read_csv(data_path)
        # Keep only relevant columns and drop nulls
        if 'review/text' in self.reviews_df.columns:
            required_cols = ['Id', 'Title', 'review/text', 'review/summary', 'review/score']
            # Only keep columns that exist
            available_cols = [col for col in required_cols if col in self.reviews_df.columns]
            self.reviews_df = self.reviews_df[available_cols].dropna(subset=['review/text'])
        print(f"Loaded {len(self.reviews_df)} reviews")
    
    def create_embeddings(self, batch_size: int = 100) -> np.ndarray:
        """
        Create embeddings for all review texts
        
        Args:
            batch_size: Number of texts to embed in each batch
            
        Returns:
            numpy array of embeddings
        """
        if self.reviews_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        texts = self.reviews_df['review/text'].astype(str).tolist()
        embeddings = []
        
        print(f"Creating embeddings for {len(texts)} reviews...")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                if (i + batch_size) % 500 == 0:
                    print(f"Processed {min(i + batch_size, len(texts))} / {len(texts)} reviews...")
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Continue with next batch
                continue
        
        self.embeddings = np.array(embeddings).astype('float32')
        print(f"Created {len(self.embeddings)} embeddings with dimension {self.embeddings.shape[1]}")
        return self.embeddings
    
    def build_index(self, embeddings: np.ndarray = None):
        """
        Build FAISS index for efficient similarity search
        
        Args:
            embeddings: Optional embeddings array. If None, uses self.embeddings
        """
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("No embeddings available. Call create_embeddings() first.")
            embeddings = self.embeddings
        
        dimension = embeddings.shape[1]
        # Use L2 distance (Euclidean) - FAISS uses Inner Product by default, so we normalize
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        self.index.add(embeddings)
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def save_index(self, index_path: str, embeddings_path: str = None):
        """Save FAISS index and optionally embeddings to disk"""
        if self.index is None:
            raise ValueError("No index built. Call build_index() first.")
        
        faiss.write_index(self.index, index_path)
        print(f"Saved index to {index_path}")
        
        if embeddings_path and self.embeddings is not None:
            np.save(embeddings_path, self.embeddings)
            print(f"Saved embeddings to {embeddings_path}")
    
    def load_index(self, index_path: str, embeddings_path: str = None):
        """Load FAISS index and optionally embeddings from disk"""
        self.index = faiss.read_index(index_path)
        print(f"Loaded index with {self.index.ntotal} vectors from {index_path}")
        
        if embeddings_path and os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            print(f"Loaded embeddings from {embeddings_path}")
    
    def retrieve_relevant_reviews(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve most relevant reviews for a query
        
        Args:
            query: User query string
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with review information and similarity scores
        """
        if self.index is None:
            raise ValueError("No index loaded. Call build_index() or load_index() first.")
        
        # Create embedding for query
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
        except Exception as e:
            raise ValueError(f"Error creating query embedding: {e}")
        
        # Search in index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get relevant reviews
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.reviews_df):
                review = self.reviews_df.iloc[idx].to_dict()
                review['similarity_score'] = float(distance)
                review['rank'] = i + 1
                results.append(review)
        
        return results
    
    def generate_response(self, query: str, top_k: int = 5) -> str:
        """
        Generate a response using RAG (non-streaming version)
        
        Args:
            query: User query
            top_k: Number of relevant reviews to retrieve
            
        Returns:
            Generated response string
        """
        # Use streaming version and collect all chunks
        full_response = ""
        for chunk in self.generate_response_stream(query, top_k):
            full_response += chunk
        return full_response
    
    def generate_response_stream(self, query: str, top_k: int = 5) -> Iterator[str]:
        """
        Generate a streaming response using RAG
        
        Args:
            query: User query
            top_k: Number of relevant reviews to retrieve
            
        Yields:
            Response chunks as strings
        """
        # Retrieve relevant reviews
        relevant_reviews = self.retrieve_relevant_reviews(query, top_k)
        
        if not relevant_reviews:
            yield "I couldn't find any relevant reviews for your query."
            return
        
        # Build context from retrieved reviews
        context_parts = []
        for review in relevant_reviews:
            title = review.get('Title', 'Unknown')
            score = review.get('review/score', 'N/A')
            summary = review.get('review/summary', '')
            text = review.get('review/text', '')[:500]  # Limit text length
            
            context_parts.append(
                f"Book: {title}\n"
                f"Rating: {score}/5\n"
                f"Summary: {summary}\n"
                f"Review: {text}\n"
                f"---\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt for OpenAI
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions about books based on customer reviews. Be specific and cite information from the reviews when possible. If the reviews don't contain enough information to answer the question, say so."
            },
            {
                "role": "user",
                "content": f"""Based on the following customer reviews, please answer the user's question.

Customer Reviews:
{context}

User Question: {query}

Answer:"""
            }
        ]
        
        try:
            stream = self.openai_client.chat.completions.create(
                model=self.generation_model,
                messages=messages,
                stream=True,
                temperature=0.7
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def get_relevant_reviews_info(self, query: str, top_k: int = 5) -> List[Dict]:
        """Get information about retrieved reviews for display"""
        return self.retrieve_relevant_reviews(query, top_k)

