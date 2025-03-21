from scraper import scrape_alphabets
from embedder import combine_articles, create_embeddings, store_in_vector_db
from query_handler import medical_query_input
from utils import input_alphabet
import os
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