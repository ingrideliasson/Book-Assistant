"""
Script to build and save the FAISS index for the RAG system
Run this once to create the embeddings and index
"""

import os
import sys
from dotenv import load_dotenv

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "app"))

from rag_system import RAGSystem

# Load environment variables
load_dotenv()


def main():
    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not openai_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    if not gemini_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        return
    
    # Initialize RAG system
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reviews_9000_popular.csv")
    rag = RAGSystem(
        openai_api_key=openai_key,
        gemini_api_key=gemini_key,
        data_path=data_path
    )
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = rag.create_embeddings()
    
    # Build index
    print("Building FAISS index...")
    rag.build_index(embeddings)
    
    # Save index
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index.bin")
    embeddings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "embeddings.npy")
    
    print("Saving index...")
    rag.save_index(index_path, embeddings_path)
    
    print("✅ Index built and saved successfully!")
    print(f"Index saved to: {index_path}")
    print(f"Embeddings saved to: {embeddings_path}")


if __name__ == "__main__":
    main()

