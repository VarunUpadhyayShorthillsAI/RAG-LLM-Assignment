import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

BASE_URL = "https://medlineplus.gov/ency/"

def fetch_page(url):
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Extract article title
    title_tag = soup.find("h1", class_="with-also", itemprop="name")
    article_title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # Remove invalid filename characters
    safe_title = "".join(c for c in article_title if c.isalnum() or c in " _-").strip()

    extracted_text = {"Title": safe_title}

    # Extract sections
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

def load_articles():
    """Load all articles"""
    articles = []
    for alpha in os.listdir("articles"):
        folder_path = os.path.join("articles", alpha)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        
                    # Parse content
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

def chunk_text(text, max_tokens=500):
    """Split text into smaller chunks based on a max token limit"""
    words = text.split()
    chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    return chunks

def create_embeddings(articles, model_name="all-MiniLM-L6-v2"):
    """Create embeddings for chunked articles"""
    model = SentenceTransformer(model_name)
    texts, metadata = [], []

    for article in articles:
        full_text = article.get("Title", "") + ". "
        for section, content in article.items():
            if not section.startswith("_"):
                full_text += f"{section}: {content} "

        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append({
                "title": article.get("Title", "Untitled"),
                "file_path": article.get("_file_path", ""),
                "alphabet": article.get("_alphabet", ""),
                "chunk_id": i
            })

    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, metadata

def store_in_vector_db(embeddings, metadata, index_path="medical_index.faiss", metadata_path="metadata.pickle"):
    """Store embeddings in a FAISS vector database"""
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

def search_vector_db(query, top_k=5, model_name="all-MiniLM-L6-v2", index_path="medical_index.faiss", metadata_path="metadata.pickle"):
    """Search the vector database with a query"""
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])[0].reshape(1, -1).astype('float32')

    index = faiss.read_index(index_path)

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):
            results.append({
                "metadata": metadata[idx],
                "distance": float(distances[0][i])
            })

    return results

def process_alphabet(alphabet):
    """Scrape and save articles for a given alphabet"""
    article_links = get_article_links(alphabet)

    if not article_links:
        print("No articles found.")
        return False
    else:
        for link in article_links:
            print(f"\nExtracting from: {link}")
            html = fetch_page(link)

            if html:
                title, extracted_text = extract_text(html)
                save_to_file(alphabet, title, extracted_text)
        return True

def create_embeddings_for_all():
    """Create embeddings for all articles and store in vector database"""
    print("Loading articles...")
    articles = load_articles()

    if not articles:
        print("No articles found. Please scrape some data first.")
        return
    
    print(f"Found {len(articles)} articles. Creating embeddings...")
    embeddings, metadata = create_embeddings(articles)

    print("Storing embeddings in vector database...")
    store_in_vector_db(embeddings, metadata)

    print("Done! Embeddings stored in medical_index.faiss and metadata in metadata.pickle")

if __name__ == "__main__":
    print("Medical Text Processing Tool")
    print("1. Scrape articles for a specific alphabet")
    print("2. Create embeddings for all articles")
    print("3. Search for a medical topic")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        alphabet = input("Enter the alphabet: ").strip().upper()
        process_alphabet(alphabet)

    elif choice == "2":
        create_embeddings_for_all()

    elif choice == "3":
        if not os.path.exists("medical_index.faiss") or not os.path.exists("metadata.pickle"):
            print("Vector database not found. Please create embeddings first (option 2).")
        else:
            query = input("Enter your medical query: ")
            results = search_vector_db(query)

            print(f"\nTop {len(results)} results for '{query}':")
            for i, result in enumerate(results):
                print(f"\n{i+1}. {result['metadata']['title']}")
                print(f"   Relevance: {1/(1+result['distance']):.2f}")
                print(f"   File: {result['metadata']['file_path']}")
