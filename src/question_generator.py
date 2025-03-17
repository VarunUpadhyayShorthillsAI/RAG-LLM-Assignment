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

# Track last processed question index
progress_file = "progress.txt"

def save_progress(index):
    with open(progress_file, "w") as f:
        f.write(str(index))

def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return int(f.read().strip())
    return 0

# Generate only context CSV
input_csv = "medical_dataset_new.csv"
output_csv = "contexts_only.csv"

qa_pairs = []
with open(input_csv, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        qa_pairs.append(row)

last_processed_index = load_progress()

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["filename", "question", "context"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i, row in enumerate(qa_pairs[last_processed_index:], start=last_processed_index):
        try:
            question = row["question"]
            query_embedding = embedding_model.encode([question])
            D, I = index.search(np.array(query_embedding).astype('float32'), k=5)
            context = "\n\n".join([chunks[i] for i in I[0] if i != -1])

            writer.writerow({
                "filename": row["filename"],
                "question": question,
                "context": context
            })
            save_progress(i + 1)
        except Exception as e:
            print(f"Error processing question at index {i}: {e}")
            break

print(f"Context CSV saved as {output_csv}")
