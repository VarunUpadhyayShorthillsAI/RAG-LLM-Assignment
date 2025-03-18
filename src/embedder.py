import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def combine_articles(data_dir="articles"):
    """Combines all text from scraped articles in specified directory into one large text file."""
    combined_text = ""

    # Iterate through each alphabet folder
    for alpha in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, alpha)
        if os.path.isdir(folder_path):
            # Process each file in alphabetical order
            for file_name in sorted(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    # Append content with separator
                    combined_text += file.read() + "\n\n"

    return combined_text

def chunk_text(text, max_tokens=200):
    """Splits text into chunks based on max token limit."""
    # Simple word-based chunking approach
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def create_embeddings(text, model_name="all-MiniLM-L6-v2"):
    """Generates embeddings for the combined text chunks."""
    # Load sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Split text into manageable chunks
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    # Generate vector embeddings for each chunk
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, chunks

def store_in_vector_db(embeddings, chunks, index_path="medical_index.faiss", metadata_path="metadata.pickle"):
    """Stores embeddings and corresponding text in FAISS."""
    # Convert to numpy array with correct data type
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]

    # Create FAISS index using L2 distance
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, index_path)

    # Save text chunks as metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(chunks, f)

    print("Embeddings stored successfully!")