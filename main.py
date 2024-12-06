import streamlit as st
import asyncio

from dotenv import load_dotenv

# from utils.contextual_query import query_documents
from utils.fastapi_integration import fastapi_upload_files, query_documents
from utils.milvus_doc_insert import load_uploaded_documents, temp_write_uploaded_file
import os

import warnings
warnings.filterwarnings("ignore")


## Experiment tracking
from langsmith import traceable
load_dotenv(dotenv_path=".env", override=True)


    
collection_name = "multi_file_collection"
embedder_model="all-MiniLM-L6-v2"


def construct_prompt_with_memory(messages, current_query, max_tokens=2048):
    """
    Construct a prompt for the LLM with chat history and the current query.
    """
    history = ""
    for msg in messages:
        history += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # Truncate if history exceeds max tokens
    tokens = history + f"User: {current_query}\nAssistant:"
    if len(tokens.split()) > max_tokens:
        truncated_history = "...\n".join(history.split("\n")[-(max_tokens // 2):])
        tokens = truncated_history + f"\nUser: {current_query}\nAssistant:"

    return tokens



# Streamlit UI

# Streamlit Page Config
st.set_page_config(page_title="DocSensei: LLaMA2 based Contextual Chatbot🦙💬", layout="centered")

custom_css = """
<style>
    .main {
        max-width: 1000px; /* Adjust width as desired */
        margin: 0 auto;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(
    "<h2 style='text-align: center;'>DocSensei: LLaMA2-Based Contextual Chatbot 🦙💬</h2>",
    unsafe_allow_html=True,
)
# Title and Description
# st.title("DocSensei: LLaMA2 based Contextual Chatbot🦙💬")
st.write(
    "This chatbot allows you to upload multiple files and query them using the local LLaMA2 model with Ollama."
)

# Sidebar parameters
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.01, 1.0, 0.5, 0.01)
    top_p = st.slider("Top_p", 0.01, 1.0, 0.9, 0.01)
    max_length = st.slider("Max_length", 20, 80, 50, 5)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]




# File Uploader
files = st.file_uploader(
    "Upload your documents (PDF, CSV, TXT):",
    type=["pdf", "csv", "txt", "md"],
    accept_multiple_files=True,
)

# uploaded_files = temp_write_uploaded_file(files)
# load_uploaded_documents(uploaded_files, collection_name, embedder_model)
uploaded_files = []
for file in files:
    file_path = asyncio.run(fastapi_upload_files(collection_name, file))
    uploaded_files.append(file_path)

st.markdown("---")  # Separator for better organization



# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


# Chat Input and Query Handling
if prompt := st.chat_input("Type your query here:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if uploaded_files:
        with st.chat_message("assistant"):
            with st.spinner("Searching through uploaded documents..."):
                
                # prompt_with_memory = construct_prompt_with_memory(
                #     st.session_state.messages, prompt
                # )
                response = asyncio.run(query_documents(prompt, collection_name))
                
                st.write(f"Answer: {response.json()["response"]}")
                
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
    else:
        st.warning("No documents uploaded to query.")
