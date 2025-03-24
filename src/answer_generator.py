import os
import csv
import pickle
import pandas as pd
import time
from tqdm import tqdm  # For progress bar
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Hardcoded file paths
INPUT_CSV_PATH = "questions_only.csv"  # Change this to your input CSV file path
OUTPUT_CSV_PATH = "questions_answers_context.csv"   # Change this to your output CSV file path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants for rate limiting and retries
RATE_LIMIT_DELAY = 1  # Delay in seconds between API calls
MAX_RETRIES = 3  # Maximum number of retries for a failed request
RETRY_DELAY = 5  # Delay in seconds before retrying after a failure

def load_questions_from_csv(input_csv_path):
    """
    Load questions from a CSV file
    Expected format: filename,question
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV file {input_csv_path} not found")
    
    questions = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:  # Ensure we have both filename and question
                filename = row[0].strip()
                question = row[1].strip()
                questions.append((filename, question))
    
    return questions

def get_document_content(filename):
    """Read and return the content of a specific document"""
    if not os.path.exists(filename):
        return f"ERROR: File {filename} not found"
    
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def initialize_mistral_model():
    """Initializes the Mistral model using LangChain."""
    # Load API key from environment variables
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")
    
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.2,  # Lower temperature for more factual responses
        max_retries=2,    # Retry on API failures
        api_key=mistral_api_key
    )
    return llm

def load_vectorstore():
    """Load the cached vectorstore"""
    if not os.path.exists("vectorstore.pkl"):
        raise FileNotFoundError("Vectorstore not found. Please run your main script to create it first.")
    
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    
    return vectorstore

def process_questions_and_save_results():
    """
    Main function to process questions and save results
    """
    # Load questions from CSV
    print(f"Loading questions from {INPUT_CSV_PATH}...")
    questions = load_questions_from_csv(INPUT_CSV_PATH)
    print(f"Loaded {len(questions)} questions")
    
    # Load the vectorstore
    print("Loading vectorstore...")
    try:
        vectorstore = load_vectorstore()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Initialize the RAG pipeline
    print("Initializing RAG pipeline...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    llm = initialize_mistral_model()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a medical assistant. Answer the user's question based ONLY on the provided context.
    If the answer cannot be found in the context, say "I don't have enough information to answer that question."
    Always explain medical terms in simple language and be thorough in your answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """)
    
    # Format docs function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Process each question and save results
    results = []
    
    print("Processing questions...")
    for i, (filename, question) in enumerate(tqdm(questions, desc="Processing", unit="question")):
        retries = 0
        while retries <= MAX_RETRIES:
            try:
                # Get the document content (for context reference)
                document_content = get_document_content(filename)
                
                # Generate answer using RAG pipeline
                answer = rag_chain.invoke(question)
                
                # Get the retrieved context
                retrieved_docs = retriever.invoke(question)  # Use invoke instead of get_relevant_documents
                context = format_docs(retrieved_docs)
                
                # Add to results
                results.append({
                    "filename": filename,
                    "question": question,
                    "answer": answer,
                    "context": context
                })
                
                # Print the question and answer to the terminal
                print(f"\nQuestion: {question}")
                print(f"Answer: {answer}")
                print("-" * 80)  # Separator for readability
                
                break  # Exit retry loop on success
            
            except Exception as e:
                retries += 1
                if retries > MAX_RETRIES:
                    print(f"\n  ✗ Max retries exceeded for question: {question}")
                    results.append({
                        "filename": filename,
                        "question": question,
                        "answer": f"ERROR: {str(e)}",
                        "context": "Error retrieving context"
                    })
                    break
                else:
                    print(f"\n  ✗ Error processing question (retry {retries}/{MAX_RETRIES}): {e}")
                    time.sleep(RETRY_DELAY)  # Wait before retrying
        
        # Sleep to avoid hitting rate limits
        time.sleep(RATE_LIMIT_DELAY)
    
    # Save results to CSV
    print(f"\nSaving results to {OUTPUT_CSV_PATH}...")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"Successfully processed {len(results)} questions and saved results to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    process_questions_and_save_results()