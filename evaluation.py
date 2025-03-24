import os
import pandas as pd
import json
import time
import requests
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from tqdm import tqdm
from dotenv import load_dotenv
from bert_score import score as bert_score

# Load API keys and environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
RESULTS_FILE = "final_evaluation_results.csv"
TEST_CASES_FILE = "questions_answers_context.csv"
MISTRAL_MODEL = "mistral-tiny"

# Metric weights for final score calculation
METRIC_WEIGHTS = {
    "rouge_score": 0.1,
    "cosine_similarity": 0.4,
    "bert_score_f1": 0.5
}

# Initialize embedding models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def calculate_metrics(generated, reference):
    """Calculate evaluation metrics."""
    if not generated or not reference:
        return {key: np.nan for key in METRIC_WEIGHTS.keys() | {"bert_score_precision", "bert_score_recall"}}
    
    emb_gen = similarity_model.encode(generated)
    emb_ref = similarity_model.encode(reference)
    cosine_sim = np.dot(emb_gen, emb_ref) / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref))
    rouge_score_value = rouge.score(reference, generated)['rougeL'].fmeasure
    bert_precision, bert_recall, bert_f1 = bert_score([generated], [reference], lang="en", model_type="bert-base-uncased")
    
    metrics = {
        "rouge_score": rouge_score_value,
        "cosine_similarity": float(cosine_sim),
        "bert_score_precision": bert_precision.mean().item(),
        "bert_score_recall": bert_recall.mean().item(),
        "bert_score_f1": bert_f1.mean().item(),
    }
    metrics["final_score"] = sum(METRIC_WEIGHTS[k] * metrics[k] for k in METRIC_WEIGHTS)
    return metrics

def load_test_cases(filepath):
    """Load test cases from CSV."""
    try:
        df = pd.read_csv(filepath)
        df.rename(columns=lambda x: x.lower().strip(), inplace=True)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

def qa_pipeline(question, context=""):
    """Query Mistral API."""
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"].strip() if response.status_code == 200 else "Error generating response"

def process_test_cases():
    """Run test cases and save results."""
    df = load_test_cases(TEST_CASES_FILE)
    if df.empty:
        print("No test cases found.")
        return
    
    results = []
    
    # Progress bar for all questions
    pbar = tqdm(total=len(df), desc="Processing")  # Show total number of iterations as per DataFrame length
    for i, row in df.iterrows():
        generated = qa_pipeline(row["question"], row["context"])
        metrics = calculate_metrics(generated, row["answer"])
        
        results.append({
            "question": row["question"],
            "context": row["context"],
            "generated_answer": generated,
            "reference_answer": row["answer"],
            **metrics
        })
        
        pbar.update(1)
    
    pbar.close()
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    process_test_cases()
