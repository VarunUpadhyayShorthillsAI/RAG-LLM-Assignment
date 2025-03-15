import os
import re
import csv
import time
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_mistralai import ChatMistralAI
import getpass
from tenacity import retry, stop_after_attempt, wait_exponential  # Import retry functionality

# Set Mistral API key
os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

# Initialize Mistral model
mistral_llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.2,  # Lower temperature for more factual responses
    max_retries=2,    # Retry on API failures
)

# Initialize FAISS index and metadata
index_path = "medical_index.faiss"
metadata_path = "metadata.pickle"
index = faiss.read_index(index_path)
with open(metadata_path, 'rb') as f:
    chunks = pickle.load(f)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_questions(text):
    """Generates questions from the text of a single file."""
    title_match = re.search(r"Title: (.+)", text)
    title = title_match.group(1).strip() if title_match else "Unknown Disease"

    # Define valid keywords (only Symptoms, Treatment, and Prevention)
    valid_keywords = ["Symptoms", "Treatment", "Prevention"]

    # Extract keywords
    keywords = re.findall(r"\n([A-Z][a-zA-Z ]+?)\n", text)
    keywords = [kw for kw in keywords if kw in valid_keywords]  # Filter valid keywords

    # Generate questions with proper grammar
    questions = []
    for keyword in keywords:
        if keyword == "Symptoms":
            questions.append(f"What are the symptoms of {title}?")
        elif keyword == "Treatment":
            questions.append(f"What is the treatment for {title}?")
        elif keyword == "Prevention":
            questions.append(f"How can {title} be prevented?")

    return questions


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_answer(question, context):
    """Generates an answer using Mistral with retry logic."""
    prompt = (
        "You are a medical assistant. Answer the user's question using ONLY the provided context. "
        "If unsure, say so. Always explain medical terms in simple language.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}"
    )

    response = mistral_llm.invoke(prompt)
    return response.content

def process_file(file_path, output_file="qa_pairs_s_to_z.csv"):
    """Processes a single file and generates QA pairs."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        questions = generate_questions(text)

        # Generate answers for each question
        for question in questions:
            try:
                # Retrieve relevant chunks from FAISS
                query_embedding = embedding_model.encode([question])
                D, I = index.search(np.array(query_embedding).astype('float32'), k=5)  # Get top 5 results
                context = "\n\n".join([chunks[i] for i in I[0] if i != -1])

                # Generate answer using Mistral
                answer = generate_answer(question, context)

                # Save QA pair to CSV
                with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=["filename", "question", "answer"])
                    writer.writerow({
                        "filename": file_path,
                        "question": question,
                        "answer": answer
                    })

                print(f"Generated answer for: {question}")
            except Exception as e:
                print(f"Error generating answer for: {question}\nError: {e}")
                raise  # Re-raise the exception to stop processing

            # Add a delay between API requests to avoid rate limiting
            time.sleep(1)  # Adjust the delay as needed

def process_folder(folder_path, output_file="qa_pairs_s_to_z.csv"):
    """Processes all files in a folder and generates QA pairs."""
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file {i + 1}/{len(files)}: {file_path}")
        process_file(file_path, output_file)

def process_folders_s_to_z(base_folder, output_file="qa_pairs_s_to_z.csv"):
    """Processes folders from S to Z inside the base folder."""
    folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    folders = [f for f in sorted(folders) if f >= 'S']  # Only process folders starting from S to Z

    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        print(f"\nProcessing folder: {folder_path}")
        process_folder(folder_path, output_file)

# Example usage
if __name__ == "__main__":
    base_folder = "./articles"  # Replace with your base folder path
    output_file = "qa_pairs_s_to_z.csv"

    # Create or clear the output file
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "question", "answer"])
        writer.writeheader()

    # Process folders from S to Z
    process_folders_s_to_z(base_folder, output_file)