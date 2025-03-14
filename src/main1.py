import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
from transformers import pipeline

# --- LLaMA 3.1 Integration ---
def initialize_llama_model():
    """Initializes the LLaMA 3.1 8B Instruct model using transformers."""
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama_pipeline = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return llama_pipeline


def generate_llama_response(query, context, llama_pipeline):
    """Enhanced generation with context validation"""
    system_prompt = """You're a medical AI assistant. Follow these rules:
1. Answer ONLY using the provided context
2. If context is insufficient, state "I cannot provide a definitive answer"
3. Highlight uncertainties
4. Use markdown formatting for clarity
5. Cite context numbers like [1], [2] etc."""

    user_prompt = f"""**Context:**\n{context}\n\n**Question:** {query}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    outputs = llama_pipeline(
        messages,
        max_new_tokens=512,
        temperature=0.3,  # Lower for medical accuracy
        top_p=0.85,
        repetition_penalty=1.2,
        do_sample=True,
    )

    return postprocess_answer(outputs[0]["generated_text"][-1]["content"])

def postprocess_answer(text):
    """Clean and validate responses"""
    # Remove any markdown formatting
    text = re.sub(r'\*+', '', text)
    
    # Ensure citations reference valid context numbers
    text = re.sub(r'\[(\d+)\]', lambda m: f"[Source {m.group(1)}]" 
                  if m.group(1).isdigit() else m.group(0), text)
    
    # Truncate if too long
    return text[:2000] + "..." if len(text) > 2000 else text
# --- Core Functions ---
def fetch_page(url):
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract article title
    title_tag = soup.find("h1", class_="with-also", itemprop="name")
    article_title = title_tag.get_text(strip=True) if title_tag else "Untitled"
    safe_title = "".join(c for c in article_title if c.isalnum() or c in " _-").strip()

    extracted_text = [f"Title: {safe_title}"]

    for section in soup.find_all("div", class_="section"):
        title_div = section.find("div", class_="section-title")
        body_div = section.find("div", class_="section-body")

        if title_div and body_div:
            section_title = title_div.get_text(strip=True)
            section_content = body_div.get_text(" ", strip=True)
            
            if any(exclude in section_title.lower() for exclude in ["images", "references", "review date"]):
                continue

            extracted_text.append(f"\n{section_title}\n{section_content}")

    return safe_title, "\n".join(extracted_text)

def save_to_file(alphabet, title, content):
    folder_path = os.path.join("articles", alphabet)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{title}.txt")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"Saved: {file_path}")

def get_article_links(alphabet):
    url = f"{BASE_URL}encyclopedia_{alphabet}.htm"
    html = fetch_page(url)
    
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    article_links = []

    for li in soup.select("#mplus-content li"):
        if not li.get("class"):
            a_tag = li.find("a", href=True)
            if a_tag and a_tag["href"].startswith("article/"):
                article_links.append(BASE_URL + a_tag["href"])

    return article_links

def scrape_alphabets(alphabets):
    for alphabet in alphabets:
        print(f"\nProcessing articles for: {alphabet}")
        article_links = get_article_links(alphabet)

        for link in article_links:
            print(f"Extracting from: {link}")
            html = fetch_page(link)

            if html:
                title, extracted_text = extract_text(html)
                save_to_file(alphabet, title, extracted_text)

def combine_articles(data_dir="articles"):
    """Combines all text from scraped articles in specified directory into one large text file."""
    combined_text = ""

    for alpha in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, alpha)
        if os.path.isdir(folder_path):
            for file_name in sorted(os.listdir(folder_path)):  # Ensure alphabetical order
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    combined_text += file.read() + "\n\n"

    return combined_text

def chunk_text(text, max_tokens=200):
    """Splits text into chunks based on max token limit."""
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]


def create_embeddings(text, model_name="all-MiniLM-L6-v2"):
    """Generates embeddings for the combined text chunks."""
    model = SentenceTransformer(model_name)
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, chunks

def store_in_vector_db(embeddings, chunks, index_path="medical_index.faiss", metadata_path="metadata.pickle"):
    """Stores embeddings and corresponding text in FAISS."""
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(metadata_path, 'wb') as f:
        pickle.dump(chunks, f)

    print("Embeddings stored successfully!")

def input_alphabet():
    """Prompts the user to input an alphabet for scraping."""
    alphabet = input("Enter the alphabet to scrape (e.g., A, B, C) or 'ALL' for all alphabets: ").strip().upper()
    return alphabet

def medical_query_input(query, index_path="medical_index.faiss", top_k=5):
    """Improved retrieval with score thresholding"""
    index = faiss.read_index(index_path)
    with open("metadata.pickle", 'rb') as f:
        metadata = pickle.load(f)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    
    # Get scores and indices
    scores, indices = index.search(query_embedding, top_k*2)  # Get extra for filtering
    
    # Apply score threshold (0.5 is arbitrary - adjust based on your data)
    valid_indices = [i for i, score in zip(indices[0], scores[0]) if score < 0.5]
    
    # Combine best chunks with context preservation
    context_chunks = [metadata["chunks"][i] for i in valid_indices[:top_k]]
    context = "\n\n".join([
        f"Context {i+1}: {chunk}" 
        for i, chunk in enumerate(context_chunks)
    ])
    
    # Generate answer
    llama_pipeline = initialize_llama_model()
    answer = generate_llama_response(query, context, llama_pipeline)
    
    # Display results with scores
    print("\n=== Generated Answer ===")
    print(answer)
    print("\n=== Supporting Context ===")
    for i, chunk in enumerate(context_chunks):
        print(f"\n[Relevance Score: {scores[0][i]:.2f}]\n{chunk[:200]}...")
        
# --- Menu Functions ---
def scrape_option():
    """Function to handle scraping option."""
    alphabet_to_scrape = input_alphabet()
    
    if alphabet_to_scrape == 'ALL':
        # Scrape all alphabets from A to Z
        alphabets = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        scrape_alphabets(alphabets)
    else:
        # Scrape the specified alphabet
        scrape_alphabets([alphabet_to_scrape])
    
    print("Scraping completed successfully!")

def embedding_option():
    """Function to handle embedding option for all scraped data."""
    data_dir = input("Enter the directory where scraped data is stored (default: 'articles'): ").strip()
    if not data_dir:
        data_dir = "articles"
        
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return
        
    print("\nCombining all articles from all alphabets...")
    combined_text = combine_articles(data_dir)
    
    if not combined_text:
        print("No articles found! Please scrape some data first.")
        return
        
    print("\nGenerating embeddings for all data...")
    embeddings, chunks = create_embeddings(combined_text)
    
    print("\nStoring in FAISS...")
    store_in_vector_db(embeddings, chunks)
    
    print("Embedding process completed for all available data!")

def query_option():
    """Function to handle medical query option."""
    if not (os.path.exists("medical_index.faiss") and os.path.exists("metadata.pickle")):
        print("Error: Embeddings not found! Please create embeddings first.")
        return
        
    medical_query = input("\nEnter your medical question: ")
    medical_query_input(medical_query)

# --- Main Program ---
if __name__ == "__main__":
    while True:
        print("\n--- Medical Information System ---")
        print("1. Scrape data")
        print("2. Create embeddings from all scraped data")
        print("3. Make a medical query")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            scrape_option()
        elif choice == "2":
            embedding_option()
        elif choice == "3":
            query_option()
        elif choice == "4":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")