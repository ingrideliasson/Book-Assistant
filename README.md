# AI Book Assistant - RAG Application

A Retrieval-Augmented Generation (RAG) application that answers questions about books based on real customer reviews from Amazon. It uses a RAG pipeline combining OpenAI embeddings, FAISS vector search, and GPT-4o for natural-language generation.

## Features

- **RAG System**: Uses OpenAI API for embeddings and for intelligent question answering
- **Vector Search**: FAISS for efficient similarity search across reviews
- **Streamlit UI**: User-friendly web interface for querying the system

Users can ask questions like:

“What do readers like about Stephen King's books?”
“Which books have the most positive reviews?”
“What do people say about science-fiction novels?”

The app retrieves the most relevant reviews from thousands of entries and generates a contextual, review-grounded answer. The user can see which reviews were retrieved as context for the answer.

## Demo

https://github.com/user-attachments/assets/f6e89cc2-8ba7-46be-b4f2-273206773e78

## Data
I used a subset of an Amazon book review dataset which originally contained 3,000,000 reviews. I narrowed them down to 9,000 by extracting the 3,000 most popular books (with the most reviews), and use 3 reviews for each books. This makes the dataset broad with many unique books, while still offering depth with several reviews per book. See the data_exploration.ipynb notebook for the data cleaning process, and some data insights. 

## Local use

### 1. Clone the repo
```bash
git clone https://github.com/ingrideliasson/Book-Assistant.git
```

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
```

**Getting API Keys:**
- **OpenAI**: Sign up at https://platform.openai.com/ and get your API key

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

### 4. Run the Application

```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`
