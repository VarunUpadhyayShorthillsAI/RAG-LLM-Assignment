from scraper import scrape_alphabets
from embedder import combine_articles, create_embeddings, store_in_vector_db
from query_handler import medical_query_input
from utils import input_alphabet
import os

def scrape_option():
    """Function to handle scraping option."""
    # Get user input for which alphabet to scrape
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
    # Get directory path from user or use default
    data_dir = input("Enter the directory where scraped data is stored (default: 'articles'): ").strip()
    if not data_dir:
        data_dir = "articles"
        
    # Validate directory exists before proceeding
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return
        
    print("\nCombining all articles from all alphabets...")
    combined_text = combine_articles(data_dir)
    
    # Check if any data was found
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
    # Check if necessary files exist before allowing queries
    if not (os.path.exists("medical_index.faiss") and os.path.exists("metadata.pickle")):
        print("Error: Embeddings not found! Please create embeddings first.")
        return
        
    # Get user's medical question and process it
    medical_query = input("\nEnter your medical question: ")
    medical_query_input(medical_query)