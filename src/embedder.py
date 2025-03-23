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

def chunk_text(text, max_tokens=1000, overlap=200):
    """
    Splits text into chunks based on max token limit with specified overlap.
    
    Args:
        text: The text to chunk
        max_tokens: Maximum number of tokens (words) per chunk
        overlap: Number of tokens to overlap between chunks
    
    Returns:
        List of text chunks with overlap
    """
    words = text.split()
    chunks = []
    
    # Handle empty text case
    if not words:
        return chunks
    
    # Calculate stride (step size between chunks)
    stride = max_tokens - overlap
    
    # Ensure stride is at least 1 to prevent infinite loop
    stride = max(1, stride)
    
    # Create overlapping chunks
    for i in range(0, len(words), stride):
        # Take max_tokens words or whatever is left
        chunk = words[i:i + max_tokens]
        if chunk:  # Only add non-empty chunks
            chunks.append(" ".join(chunk))
    
    return chunks

def create_embeddings(text, model_name="all-MiniLM-L6-v2"):
    """Generates embeddings for the combined text chunks with metadata."""
    # Load sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Split text into manageable chunks with overlap
    chunks = chunk_text(text, max_tokens=200, overlap=50)
    
    print(f"Total chunks: {len(chunks)}")
    
    # Generate vector embeddings for each chunk
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Create metadata with chunk index and content
    metadata = [{"index": i, "content": chunk} for i, chunk in enumerate(chunks)]
    
    return embeddings, metadata

def store_in_vector_db(embeddings, metadata, index_path="medical_index.faiss", metadata_path="metadata.pickle"):
    """Stores embeddings and corresponding metadata in FAISS."""
    # Convert to numpy array with correct data type
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]

    # Create FAISS index using L2 distance
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, index_path)

    # Save metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print("Embeddings stored successfully!")