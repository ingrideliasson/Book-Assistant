"""
Streamlit UI for Product Review RAG Application
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Add app directory to path for imports
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

from rag_system import RAGSystem

# Load environment variables
load_dotenv()


def initialize_rag_system():
    """Initialize RAG system with API keys"""
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        st.error("OPENAI_API_KEY not found in environment variables. Please set it in .env file or environment.")
        st.info("Create a `.env` file in the project root with: `OPENAI_API_KEY=your_key_here`")
        return None
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reviews_9000_popular.csv")
    
    if not os.path.exists(data_path):
        st.error(f"Data file not found: {data_path}")
        st.info("Make sure the reviews_9000_popular.csv file exists in the data/ folder")
        return None
    
    try:
        return RAGSystem(
            openai_api_key=openai_key,
            data_path=data_path
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Book Review Assistant",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for muted color palette and better typography
    st.markdown("""
    <style>
        /* Import a clean font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        /* Apply font to all text */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-weight: 100;
        }
        
        /* Muted color palette */
        .stApp {
            background-color: #ffffff;
        }
        
        /* Main title styling - centered */
        h1 {
            color: #2c3e50;
            font-weight: 200;
            letter-spacing: 1px;
            text-align: center;
        }
        
        /* Override Streamlit's automatic main content positioning */
        /* Reset Streamlit's automatic margin for sidebar */
        section[data-testid="stMain"] {
            margin-left: 0 !important;
            padding-left: 0 !important;
        }
        
        /* Center the main content on the full viewport */
        .main {
            margin-left: 0 !important;
            width: 100vw !important;
        }
        
        /* Center the block container on the full viewport width */
        .main .block-container {
            max-width: 800px;
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            /* Center on full viewport width, accounting for sidebar on left */
            margin-left: calc((100vw - 800px) / 2) !important;
            margin-right: auto;
            /* Adjust for sidebar: sidebar is ~21rem, so shift right by half */
            transform: translateX(10.5rem);
        }
        
        /* Sidebar positioning - keep it on the left */
        [data-testid="stSidebar"] {
            position: fixed !important;
            left: 0 !important;
            z-index: 999;
        }
        
        /* Move sidebar content down to align with input field */
        /* The title and input field are at similar vertical positions */
        [data-testid="stSidebar"] .element-container:first-of-type {
            margin-top: 8rem;
        }
        
        /* Center the Example Queries title */
        [data-testid="stSidebar"] h3 {
            text-align: center;
        }
        
        /* Style query buttons in sidebar */
        [data-testid="stSidebar"] .stButton > button {
            background-color: #e1e3e3;
            color: white;
            padding-x: 1rem;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #c9c9c9;
        }
        
        
        /* Headers */
        h2, h3, h4 {
            color: #34495e;
            font-weight: 100;
        }
        
        /* Text color */
        p, div, span {
            color: #4a5568;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            margin-left: 4rem;
        }
        
        /* Button styling - muted */
        .stButton > button {
            background-color: #5a6c7d;
            color: white;
            border: none;
            font-weight: 500;
            transition: background-color 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #4a5568;
        }

        .stButton > button[kind="primary"] {
            background-color: #e1e3e3;
            color: #2c3e50;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            width: auto;
            min-width: 120px;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: #c9c9c9;
            color: #2c3e50;
        }
        
        /* Input fields */
        .stTextInput > div > div > input {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            color: #2d3748;
        }
        
        /* Slider */
        .stSlider {
            color: #5a6c7d;
        }
        
        
        /* Expander */
        .streamlit-expanderHeader {
            color: #4a5568;
            font-weight: 500;
        }
        
        /* Info boxes - muted */
        .stAlert {
            background-color: #f7fafc;
            border-left: 4px solid #a0aec0;
        }
        
        /* Success messages */
        [data-baseweb="notification"] {
            background-color: #e6f3e6;
            color: #2d5016;
        }
        
        /* Warning messages */
        [data-baseweb="notification"].warning {
            background-color: #fff4e6;
            color: #856404;
        }
        
        /* Error messages */
        [data-baseweb="notification"].error {
            background-color: #ffe6e6;
            color: #721c24;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Book Review Assistant")
    st.markdown("<p style='text-align: center; color: #4a5568;'>Ask questions about books based on customer reviews.</p>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = initialize_rag_system()
            if st.session_state.rag_system:
                # Try to load existing index
                index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index.bin")
                embeddings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "embeddings.npy")
                
                if os.path.exists(index_path):
                    try:
                        st.session_state.rag_system.load_index(index_path, embeddings_path)
                        st.session_state.index_loaded = True
                    except Exception as e:
                        st.warning(f"Could not load existing index: {e}")
                        st.session_state.index_loaded = False
                else:
                    st.session_state.index_loaded = False
    
    if st.session_state.rag_system is None:
        st.stop()
    
    # Sidebar - only example queries
    with st.sidebar:
        st.markdown("<h3 style='text-align: center;'>Example Queries</h3>", unsafe_allow_html=True)
        example_queries = [
            "What do people like about Stephen King's books?",
            "What do people think about The Catcher in the Rye?",
            "What do reviewers say about the science fiction genre?",
            "Which are the best rated books in the romance genre?",
            "I like realism and drama, which book should I read?"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query}", use_container_width=True):
                st.session_state.query = query
                st.rerun()
    
    # Main content area - handle index loading silently
    if not st.session_state.index_loaded:
        # Try to build index automatically if it doesn't exist
        index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index.bin")
        if not os.path.exists(index_path):
            st.warning("No embeddings index found. Building index...")
            with st.spinner("Creating embeddings and building index (this may take a few minutes)..."):
                try:
                    embeddings = st.session_state.rag_system.create_embeddings()
                    st.session_state.rag_system.build_index(embeddings)
                    
                    # Save index
                    embeddings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "embeddings.npy")
                    st.session_state.rag_system.save_index(index_path, embeddings_path)
                    st.session_state.index_loaded = True
                    st.success("Index built successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error building index: {e}")
                    st.stop()
        else:
            st.stop()
    
    # Query input
    query = st.text_input(
        "",
        value=st.session_state.get('query', ''),
        placeholder="e.g., What do people like about Stephen King's books?",
        key="query_input"
    )
    
    top_k = st.slider("Number of reviews to retrieve:", min_value=3, max_value=10, value=5)
    
    # Search button - left aligned
    search_clicked = st.button("Search", type="primary", use_container_width=False)
    
    if search_clicked or query:
        if query:
            # Display query
            st.markdown(f"### Question: {query}")
            
            # Generate response with streaming
            try:
                st.markdown("### Answer:")
                # Create a placeholder for streaming response
                response_placeholder = st.empty()
                
                # Collect streaming chunks
                full_response = ""
                with st.spinner("Searching reviews..."):
                    # Retrieve reviews first (this happens before streaming)
                    relevant_reviews = st.session_state.rag_system.get_relevant_reviews_info(query, top_k)
                
                # Stream the response
                for chunk in st.session_state.rag_system.generate_response_stream(query, top_k):
                    full_response += chunk
                    # Update the placeholder with accumulated text
                    response_placeholder.markdown(full_response)
                
                # Show relevant reviews
                with st.expander("View Relevant Reviews Used", expanded=False):
                    for i, review in enumerate(relevant_reviews, 1):
                        st.markdown(f"#### Review {i} (Similarity: {review['similarity_score']:.3f})")
                        st.markdown(f"**Book:** {review.get('Title', 'Unknown')}")
                        st.markdown(f"**Rating:** {review.get('review/score', 'N/A')}/5")
                        st.markdown(f"**Summary:** {review.get('review/summary', 'N/A')}")
                        st.markdown(f"**Review Text:** {review.get('review/text', 'N/A')[:300]}...")
                        st.divider()
            
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()

