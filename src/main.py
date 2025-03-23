import os
import requests
from bs4 import BeautifulSoup
import pickle
import getpass
from typing import List, Dict

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

# Set Mistral API key
os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

BASE_URL = "https://medlineplus.gov/ency/"

# --- Web Scraping Functions ---
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

# --- LangChain RAG Pipeline Functions ---

def initialize_mistral_model():
    """Initializes the Mistral model using LangChain."""
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.2,  # Lower temperature for more factual responses
        max_retries=2,     # Retry on API failures
    )
    return llm

def create_rag_pipeline(data_dir="articles", use_cached=False):
    """Creates a complete RAG pipeline using LangChain components."""
    # Check if we have a cached vectorstore
    if use_cached and os.path.exists("vectorstore.pkl"):
        print("Loading cached vector store...")
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
        
        # Initialize embeddings model for potential new queries
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        print("Creating new vector store from documents...")
        
        # 1. Document Loading
        # Check if directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} does not exist. Please scrape data first.")
            
        # Load all text files from the articles directory
        loader = DirectoryLoader(
            data_dir, 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        
        if not documents:
            raise ValueError("No documents found! Please scrape some data first.")
        
        print(f"Loaded {len(documents)} documents")
        
        # 2. Text Splitting with Increased Chunk Size and Significant Overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased from 500 to 1500 as requested
            chunk_overlap=200,  # Significant overlap (20% of chunk size)
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"Split into {len(chunks)} chunks")
        print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0} characters")
        
        # 3. Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 4. Vector Storage
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Cache the vectorstore
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
    
    # 5. Retriever with increased k for more comprehensive context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})  # Increased from 5 to 7
    
    # 6. LLM
    llm = initialize_mistral_model()
    
    # 7. Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    You are a medical assistant. Answer the user's question based ONLY on the provided context.
    If the answer cannot be found in the context, say "I don't have enough information to answer that question."
    Always explain medical terms in simple language and be thorough in your answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """)
    
    # 8. Complete RAG Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def input_alphabet():
    """Prompts the user to input an alphabet for scraping."""
    alphabet = input("Enter the alphabet to scrape (e.g., A, B, C) or 'ALL' for all alphabets: ").strip().upper()
    return alphabet

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
    """Function to handle creating the RAG pipeline with all scraped data."""
    data_dir = input("Enter the directory where scraped data is stored (default: 'articles'): ").strip()
    if not data_dir:
        data_dir = "articles"
        
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return
    
    try:
        # This will create and cache the vectorstore
        print("Creating LangChain RAG pipeline...")
        create_rag_pipeline(data_dir, use_cached=False)
        print("RAG pipeline created and vector store cached successfully!")
    except Exception as e:
        print(f"Error creating RAG pipeline: {e}")

def query_option():
    """Function to handle medical query option."""
    try:
        # Load the RAG pipeline (will use cached vectorstore if available)
        rag_chain = create_rag_pipeline(use_cached=True)
        
        medical_query = input("\nEnter your medical question: ")
        
        print("\nProcessing your query using LangChain RAG pipeline...")
        response = rag_chain.invoke(medical_query)
        
        print("\n=== Generated Answer ===")
        print(response)
        
    except FileNotFoundError:
        print("Error: Vector store not found! Please create embeddings first.")
    except Exception as e:
        print(f"Error processing query: {e}")

# --- Main Program ---
if __name__ == "__main__":
    while True:
        print("\n--- Medical Information System with LangChain RAG ---")
        print("1. Scrape medical data")
        print("2. Create LangChain RAG pipeline from scraped data")
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