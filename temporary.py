import os
import requests
import faiss
import pickle
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# Set API Key for Llama-2 (Hugging Face)
HF_API_KEY = "your_huggingface_api_key"  # Replace with actual API key
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BASE_URL = "https://medlineplus.gov/ency/"

# Function to fetch page HTML
def fetch_page(url):
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

# Extract text sections and chunk them
def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1", class_="with-also", itemprop="name")
    article_title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    safe_title = "".join(c for c in article_title if c.isalnum() or c in " _-").strip()
    extracted_text = {"Title": safe_title}

    for section in soup.find_all("div", class_="section"):
        title_div = section.find("div", class_="section-title")
        body_div = section.find("div", class_="section-body")

        if title_div and body_div:
            section_title = title_div.get_text(strip=True)
            section_content = body_div.get_text(" ", strip=True)

            # Exclude unwanted sections
            if any(exclude in section_title.lower() for exclude in ["images", "references", "review date"]):
                continue

            extracted_text[section_title] = section_content

    return safe_title, extracted_text

# Save extracted text into structured folder
def save_to_file(alphabet, title, content):
    folder_path = os.path.join("articles", alphabet)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{title}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        for section, text in content.items():
            file.write(f"\n{section}\n{text}\n")

    print(f"Saved: {file_path}")

# Load articles from saved files
def load_articles():
    articles = []
    for alpha in os.listdir("articles"):
        folder_path = os.path.join("articles", alpha)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                    
                    article_data = {"Title": file_name.replace(".txt", ""), "_file_path": file_path, "_alphabet": alpha, "Content": content}
                    articles.append(article_data)
    return articles

# Create embeddings for retrieved chunks
def create_embeddings(articles):
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [article["Content"] for article in articles]
    metadata = [{"title": article["Title"], "file_path": article["_file_path"], "alphabet": article["_alphabet"]} for article in articles]

    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, metadata

# Store embeddings in FAISS vector DB
def store_in_vector_db(embeddings, metadata):
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, "medical_index.faiss")
    with open("metadata.pickle", 'wb') as f:
        pickle.dump(metadata, f)

# Load FAISS index & metadata
def load_faiss_index():
    index = faiss.read_index("medical_index.faiss")
    with open("metadata.pickle", 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

# Retrieve relevant document chunks from FAISS
def retrieve_relevant_docs(query, top_k=3):
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])[0].reshape(1, -1).astype('float32')

    index, metadata = load_faiss_index()
    distances, indices = index.search(query_embedding, top_k)

    retrieved_docs = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(metadata):
            with open(metadata[idx]["file_path"], "r", encoding="utf-8") as file:
                retrieved_docs.append(file.read())

    return retrieved_docs

# Generate answer using Llama-2 API
def generate_llm_response(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs[:2])  # Use top 2 retrieved documents
    prompt = f"Answer based on these documents:\n\n{context}\n\nQuery: {query}\nAnswer:"

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_length": 512}}

    response = requests.post(f"https://api-inference.huggingface.co/models/{LLM_MODEL}", headers=headers, json=payload)
    return response.json()[0]["generated_text"] if response.status_code == 200 else "Error in LLM response"

# Full RAG-based Q&A system
def rag_query(query):
    retrieved_docs = retrieve_relevant_docs(query)
    print("\n🔹 Retrieved Docs:\n", "\n---\n".join(retrieved_docs[:2]))  # Show retrieved context
    return generate_llm_response(query, retrieved_docs)

# Run options
if __name__ == "__main__":
    print("Medical Text Processing Tool")
    print("1. Scrape articles for a specific alphabet")
    print("2. Create embeddings for all articles")
    print("3. Query the RAG system")

    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        alphabet = input("Enter the alphabet: ").strip().upper()
        process_alphabet(alphabet)
    
    elif choice == "2":
        print("Loading and embedding articles...")
        articles = load_articles()
        embeddings, metadata = create_embeddings(articles)
        store_in_vector_db(embeddings, metadata)
        print("Embeddings stored!")

    elif choice == "3":
        user_query = input("Enter your medical query: ")
        response = rag_query(user_query)
        print("\n💡 LLM Response:\n", response)
