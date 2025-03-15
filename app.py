import streamlit as st
import os
from src.query_handler import medical_query_input
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Set page title
st.set_page_config(page_title="Medical Information System")

# Initialize session state for API key
if "api_key_saved" not in st.session_state:
    st.session_state.api_key_saved = False

# Title of the app
st.title("Medical Information System")

# Sidebar for API key input
st.sidebar.header("Configuration")

# API key input section
if not st.session_state.api_key_saved:
    api_key = st.sidebar.text_input("Enter your Mistral API key:", type="password")
    if st.sidebar.button("Save API Key"):
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key
            st.session_state.api_key_saved = True
            st.sidebar.success("API key saved for this session!")
        else:
            st.sidebar.error("Please enter a valid API key")
else:
    st.sidebar.success("API key is set for this session")
    if st.sidebar.button("Change API Key"):
        st.session_state.api_key_saved = False
        st.experimental_rerun()

# Input box for user query
query = st.text_input("Enter your medical question:")

# Button to submit
if st.button("Submit"):
    if query:
        if st.session_state.api_key_saved:
            with st.spinner("Generating response..."):
                try:
                    answer, context = medical_query_input(query)
                    
                    # Display the answer
                    st.markdown("### Generated Answer")
                    st.write(answer)
                    
                    # Display the supporting context
                    with st.expander("Supporting Context"):
                        st.write(context)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please set your API key in the sidebar first.")
    else:
        st.warning("Please enter a question.")

# Display some example questions
st.markdown("### Example Questions")
st.info("""
- What are the symptoms of diabetes?
- How do I treat a common cold?
- What causes high blood pressure?
""")