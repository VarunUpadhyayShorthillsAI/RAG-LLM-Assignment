import streamlit as st
import os
from src.query_handler import medical_query_input
import torch
from dotenv import load_dotenv
from datetime import datetime
import csv

# Load environment variables
load_dotenv()
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Set Streamlit page configuration
st.set_page_config(
    page_title="Medical Information System",
    page_icon="",
    layout="centered"
)

# Custom CSS for a clean hospital-like theme
st.markdown(
    """
    <style>
        body {  }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #81c784;
        }
        .stButton>button {
            border-radius: 10px;
            padding: 12px 24px;
            background-color: #4caf50;
            color: white;
            border: none;
            font-size: 16px;
        }
        .stButton>button:hover {
          
        }
        .stMarkdown h1 {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            color: #1e88e5;
        }
        .stMarkdown h3 {
            # color: #388e3c;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<h1>Medical Encyclopedia</h1>", unsafe_allow_html=True)

# Initialize API Key in session state
if "api_key_saved" not in st.session_state:
    st.session_state.api_key_saved = False

# Fetch API key from environment
api_key = os.getenv("MISTRAL_API_KEY")
if api_key:
    os.environ["MISTRAL_API_KEY"] = api_key
    st.session_state.api_key_saved = True

# Function to log Q&A interactions
def log_qa_interaction(question, answer, log_file="qa_logs.csv"):
    """
    Logs a Q&A interaction to a CSV file.
    """
    # Create the log entry
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    }

    # Write the log entry to the CSV file
    with open(log_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=log_entry.keys())
        if file.tell() == 0:  # Write header if file is empty
            writer.writeheader()
        writer.writerow(log_entry)

# User input box
query = st.text_input("Enter your medical question:", placeholder="e.g., What are the symptoms of pneumonia?")

# Submit button
if st.button("Get Answer"):
    if query:
        if st.session_state.api_key_saved:
            with st.spinner("Retrieving medical information..."):
                try:
                    answer, context = medical_query_input(query)
                    
                    # Display the answer
                    st.markdown("### Response")
                    st.write(answer)

                    # Log the Q&A interaction
                    log_qa_interaction(question=query, answer=answer)

                    # Show context if available
                    if context:
                        st.markdown("### Additional Information")
                        st.write(context)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please set your API key as an environment variable.")
    else: 
        st.warning("Please enter a question.")

# Example Questions section
st.markdown("### Example Questions")
st.info("""
- What are the symptoms of Anemia caused by low iron - infants and toddlers?
- What is the treatment for Anemia caused by low iron - infants and toddlers?
- How can Anemia caused by low iron - infants and toddlers be prevented?
- What are the symptoms of Alcohol use disorder?
- What is the treatment for Alcohol use disorder?
- How can Alcohol use disorder be prevented?
- What are the symptoms of Amblyopia?
""")