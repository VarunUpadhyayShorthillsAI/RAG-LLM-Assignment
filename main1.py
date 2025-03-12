import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

BASE_URL = "https://medlineplus.gov/ency/"
INDEX_PATH = "medical_index.faiss"
METADATA_PATH = "metadata.pickle"
MODEL_NAME = "all-MiniLM-L6-v2"

# ========================== SCRAPING ARTICLES ==========================

def fetch_page(url):
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

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

            if any(exclude in section_title.lower() for exclude in ["images", "references", "review date"]):
                continue

            extracted_text[section_title] = section_content

    return safe_title, extracted_text

def save_to_file(alphabet, title, content):
    folder_path = os.path.join("articles", alphabet)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{title}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        for section, text in content.items():
            file.write(f"\n{section}\n{text}\n")

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

def process_alphabet(alphabet):
    print(f"\nProcessing articles for: {alphabet}")
    article_links = get_article_links(alphabet)

    if not article_links:
        print("No articles found.")
        return
    
    for link in article_links:
        print(f"Extracting from: {link}")
        html = fetch_page(link)

        if html:
            title, extracted_text = extract_text(html)
            save_to_file(alphabet, title, extracted_text)

# ========================== LOADING & CHUNKING TEXT ==========================

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
                    
                    lines = content.strip().split("\n")
                    article_data = {}
                    current_section = None
                    
                    for line in lines:
                        if line and not line.startswith(" ") and line == line.strip():
                            current_section = line
                            article_data[current_section] = ""
                        elif current_section:
                            article_data[current_section] += line + " "
                    
                    article_data["_file_path"] = file_path
                    article_data["_alphabet"] = alpha
                    articles.append(article_data)
    
    return articles

def combine_text(articles):
    """Combines all text into one large document."""
    combined_text = ""
    for article in articles:
        combined_text += article.get("Title", "") + "\n"
        for section, content in article.items():
            if not section.startswith("_"):
                combined_text += f"{section}: {content}\n"
    return combined_text

def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    return chunks

# ========================== GENERATING & STORING EMBEDDINGS ==========================

def create_embeddings(text):
    model = SentenceTransformer(MODEL_NAME)
    chunks = chunk_text(text)
    
    metadata = [{"chunk_id": i} for i in range(len(chunks))]
    embeddings = model.encode(chunks, show_progress_bar=True)

    return np.array(embeddings).astype('float32'), metadata

def store_in_vector_db(embeddings, metadata):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

    print("Embeddings stored successfully!")

def generate_and_store_embeddings():
    print("Loading articles...")
    articles = load_articles()

    if not articles:
        print("No articles found. Please scrape some data first.")
        return
    
    print(f"Found {len(articles)} articles. Combining text...")
    combined_text = combine_text(articles)

    print("Generating embeddings...")
    embeddings, metadata = create_embeddings(combined_text)

    print("Storing in FAISS...")
    store_in_vector_db(embeddings, metadata)

# ========================== SEARCH FUNCTION ==========================

def load_faiss_index():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def search_medical_query(query, top_k=3):
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query]).astype('float32')

    index, metadata = load_faiss_index()
    _, indices = index.search(query_embedding, top_k)

    results = [metadata[idx] for idx in indices[0] if idx < len(metadata)]
    return results

# ========================== MAIN EXECUTION ==========================

if __name__ == "__main__":
    print("Medical Text Processing Tool")
    print("1. Scrape articles for alphabets")
    print("2. Create embeddings for all articles")
    print("3. Search for a medical topic")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        alphabets = input("Enter alphabets (e.g., X,Y,Z): ").strip().upper().split(",")
        for alphabet in alphabets:
            process_alphabet(alphabet)

    elif choice == "2":
        generate_and_store_embeddings()

    elif choice == "3":
        if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
            print("Vector database not found. Please create embeddings first (option 2).")
        else:
            query = input("Enter your medical query: ")
            results = search_medical_query(query)

            print("\nTop results:")
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}: {result}")
