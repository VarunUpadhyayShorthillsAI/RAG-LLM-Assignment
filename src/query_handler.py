import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langchain_mistralai import ChatMistralAI

def initialize_mistral_model():
    """Initializes the Mistral model using LangChain."""
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.2,  # Lower temperature for more factual responses
        max_retries=2,     # Retry on API failures
    )
    return llm

def generate_mistral_response(query, context, mistral_llm):
    """Generates an answer using Mistral."""
    prompt = (
        "You are a medical assistant. Answer the user's question using ONLY the provided context. "
        "If unsure, say so. Always explain medical terms in simple language.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}"
    )

    response = mistral_llm.invoke(prompt)
    return response.content

def medical_query_input(query, index_path="medical_index.faiss", metadata_path="metadata.pickle"):
    """Processes a medical query and retrieves relevant articles."""
    # Load the FAISS index and metadata
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        chunks = pickle.load(f)

    # Create embeddings for the query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])

    # Search for the nearest neighbors
    D, I = index.search(np.array(query_embedding).astype('float32'), k=5)  # Get top 5 results

    # Combine top chunks into context
    context = "\n\n".join([chunks[i] for i in I[0] if i != -1])

    # Generate answer using Mistral
    mistral_llm = initialize_mistral_model()
    answer = generate_mistral_response(query, context, mistral_llm)
    
    # Return the answer and context
    return answer, context[:1000] + "..."  # Show first 1000 chars of context