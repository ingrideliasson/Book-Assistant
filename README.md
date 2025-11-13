# AI Product Review Assistant - RAG Application

A Retrieval-Augmented Generation (RAG) application that answers questions about books based on customer reviews.

## Features

- **RAG System**: Uses OpenAI embeddings and Google Gemini for intelligent question answering
- **Vector Search**: FAISS for efficient similarity search across reviews
- **Streamlit UI**: User-friendly web interface for querying the system
- **Book Recommendations**: Get recommendations and insights based on customer reviews

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

**Getting API Keys:**
- **OpenAI**: Sign up at https://platform.openai.com/ and get your API key
- **Gemini**: Sign up at https://makersuite.google.com/app/apikey (free tier available)

### 3. Build the Embeddings Index

Run the script to create embeddings and build the FAISS index:

```bash
python scripts/build_index.py
```

This will:
- Load the review data
- Create embeddings using OpenAI's `text-embedding-3-small` model
- Build a FAISS index for fast similarity search
- Save the index to `data/faiss_index.bin`

**Note**: This process may take a few minutes depending on the number of reviews.

### 4. Run the Application

```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### Example Queries

- "What do people like about Margaret Atwood books?"
- "What books do you recommend in the fantasy genre?"
- "What are the best rated books?"
- "What do reviewers say about science fiction books?"

### How It Works

1. **Query Processing**: Your question is converted to an embedding using OpenAI
2. **Retrieval**: The system finds the most relevant reviews using FAISS vector search
3. **Generation**: Gemini generates an answer based on the retrieved reviews
4. **Display**: The answer and relevant reviews are shown in the UI

## Project Structure

```
ai-product-review-assistant/
├── app/
│   ├── app.py              # Streamlit UI application
│   └── rag_system.py        # RAG system implementation
├── data/
│   ├── reviews_5000.csv     # Review data
│   ├── faiss_index.bin      # FAISS index (created after building)
│   └── embeddings.npy       # Embeddings (created after building)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── scripts/
│   └── build_index.py       # Script to build embeddings index
├── .env                     # API keys (create from .env.example)
├── requirements.txt
└── README.md
```

## Technologies Used

- **OpenAI**: Text embeddings (`text-embedding-3-small`)
- **Google Gemini**: LLM for response generation
- **FAISS**: Vector similarity search
- **Streamlit**: Web UI framework
- **Pandas**: Data processing
- **NumPy**: Numerical operations

## Notes

- The application uses the first 5000 reviews from the dataset
- Embeddings are created using OpenAI's API (costs apply)
- Gemini API has a free tier with generous limits
- The index is built once and reused for all queries

## Troubleshooting

### Index Not Found
If you see "No embeddings index found", run:
```bash
python scripts/build_index.py
```

### API Key Errors
Make sure your `.env` file contains valid API keys and is in the project root.

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

