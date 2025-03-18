import os
import pandas as pd
import json
import time
import requests
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from evaluate import load  
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
import torch

# Load environment variables
load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
LOG_FILE = "qa_interactions.log"
RESULTS_FILE = "final_evaluation_new.csv"
TEST_CASES_FILE = "cleaned_file_with_context.csv"

VALID_MODELS = ["mistral-7b", "mistral-tiny", "gpt-4"]
MISTRAL_MODEL = "mistral-tiny"

if MISTRAL_MODEL not in VALID_MODELS:
    raise ValueError(f"Invalid model '{MISTRAL_MODEL}'. Choose from {VALID_MODELS}")

# Initialize models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Initialize BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Setup logger
def setup_logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,context,question,generated_answer,reference_answer,cosine_similarity,rouge_score,bert_similarity,precision,final_score\n")

def log_interaction(context, question, generated, reference, metrics):
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "context": context,
        "question": question,
        "generated_answer": generated,
        "reference_answer": reference,
        **metrics
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def calculate_bert_similarity(text1, text2):
    inputs = bert_tokenizer([text1, text2], return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
    return cosine_sim

def calculate_metrics(generated, reference):
    if not generated or not reference:
        return {
            "cosine_similarity": np.nan,
            "rouge_score": np.nan,
            "bert_similarity": np.nan,
            "precision": np.nan,
            "final_score": np.nan
        }

    # Cosine Similarity
    emb_gen = similarity_model.encode(generated)
    emb_ref = similarity_model.encode(reference)
    if np.any(np.isnan(emb_gen)) or np.any(np.isnan(emb_ref)):
        return {
            "cosine_similarity": np.nan,
            "rouge_score": np.nan,
            "bert_similarity": np.nan,
            "precision": np.nan,
            "final_score": np.nan
        }
    cosine_sim = np.dot(emb_gen, emb_ref) / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref))

    # ROUGE Score
    rouge_score = rouge.score(reference, generated)['rougeL'].fmeasure

    # BERT Similarity
    bert_sim = calculate_bert_similarity(generated, reference)

    # Precision
    gen_tokens = set(generated.split())
    ref_tokens = set(reference.split())
    true_positives = len(gen_tokens.intersection(ref_tokens))
    false_positives = len(gen_tokens - ref_tokens)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # Final Score (weighted average)
    final_score = (
        cosine_sim * 0.4 +
        rouge_score * 0.3 +
        bert_sim * 0.3
    )

    return {
        "cosine_similarity": float(cosine_sim),
        "rouge_score": rouge_score,
        "bert_similarity": bert_sim,
        "precision": precision,
        "final_score": final_score
    }

def load_test_cases(filepath):
    df = pd.read_csv(filepath)
    print(f" Loaded {len(df)} test cases.")
    return df

def qa_pipeline(question, context=""):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI assistant helping with RAG-based question answering."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }

    max_retries = 5
    backoff_time = 2

    for attempt in range(max_retries):
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()

        elif response.status_code == 429:
            wait_time = backoff_time * (2 ** attempt)
            print(f"⚠️ Rate limit hit. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

        else:
            print(f" Error: {response.status_code} - {response.text}")
            return "Error generating response"

    print("Max retries reached. Skipping question.")
    return "Error generating response"

def process_test_cases():
    setup_logger()
    df = load_test_cases(TEST_CASES_FILE)

    if df.empty:
        print("⚠️ No test cases found. Please check the file format.")
        return

    if not os.path.exists(RESULTS_FILE):
        pd.DataFrame(columns=[
            "question", "context", "generated_answer", "reference_answer",
            "cosine_similarity", "rouge_score", "bert_similarity", "precision", "final_score"
        ]).to_csv(RESULTS_FILE, index=False)

    pbar = tqdm(total=len(df), desc="Processing test cases")

    for idx, row in df.iterrows():
        try:
            generated = qa_pipeline(row["question"], row["context"])

            metrics = calculate_metrics(generated=generated, reference=row["answer"])

            log_interaction(
                context=row["context"],
                question=row["question"],
                generated=generated,
                reference=row["answer"],
                metrics=metrics
            )

            result_row = pd.DataFrame([{
                "question": row["question"],
                "context": row["context"],
                "generated_answer": generated,
                "reference_answer": row["answer"],
                **metrics
            }])

            result_row.to_csv(RESULTS_FILE, mode="a", header=False, index=False)

            pbar.update(1)
            pbar.set_postfix({"Processed": f"{idx+1}/{len(df)}", "Score": f"{metrics['final_score']:.2f}"})

        except Exception as e:
            error_msg = f"Error processing case {idx}: {str(e)}"
            print(f"{error_msg}")
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "error": error_msg,
                    "context": row.get("context", ""),
                    "question": row.get("question", "")
                }) + "\n")

    pbar.close()
    print(f"\nProcessing complete! Results saved to {RESULTS_FILE}. Logs in {LOG_FILE}.")

if __name__ == "__main__":
    process_test_cases()

    final_df = pd.read_csv(RESULTS_FILE)
    print("\nFinal Metrics Summary:")
    print(final_df.describe())

    print(final_df.isna().sum())
