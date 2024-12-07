import streamlit as st
import requests
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

## Experiment tracking
from langsmith import traceable


from utils.wrappers import (
    clear_chat_history, 
    upload_file_to_vector_db,
    query_rag_llm, 
    query_vector_db
    )




# Define global variables    
collection_name = "insightpro_collection"
embedder_model="all-MiniLM-L6-v2"
API_URL = "http://127.0.0.1:8000"





# Streamlit UI
# Streamlit Page Config
st.set_page_config(page_title="InsightPro: LLaMA based Contextual Chatbot🦙💬", layout="centered")

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
    "<h2 style='text-align: center;'>InsightPro: LLaMA Based Contextual Chatbot 🦙💬</h2>",
    unsafe_allow_html=True,
)
# Title and Description
# st.title("InsightPro: LLaMA2 based Contextual Chatbot🦙💬")
st.write(
    "This chatbot allows you to upload multiple files and query them using the local LLaMA models."
)
# Sidebar for clearing chat history, uploading docs and model tuning
with st.sidebar:
    st.sidebar.markdown(
            """
            # InsightPro  
            **LLaMA-based Contextual Chatbot**
            """,
            unsafe_allow_html=True,
        )
    

    # File Upload
    st.markdown("---")  # Separator for better organization
    st.subheader("Upload Documents")
    
    files = st.file_uploader(
        "Upload your documents (PDF, CSV, TXT):",
        type=["pdf", "csv", "txt", "md"],
        accept_multiple_files=True,
    )

    # Cache Uploaded Files
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

    # Upload New Files Only
    if files:
        new_files = [
            file for file in files if file.name not in st.session_state.uploaded_files
        ]
        if new_files:
            with st.spinner("Uploading new files..."):
                for file in new_files:
                    result = upload_file_to_vector_db(API_URL, file, collection_name)
                    if result["status"] == "success":
                        st.success(
                            f"File '{file.name}' uploaded to collection '{result['details']['collection']['_name']}'"
                        )
                        st.session_state.uploaded_files.add(file.name)
                    else:
                        st.error(f"Failed to upload {file.name}: {result['details']}")



    st.markdown("---") 
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


    st.markdown("---")
    st.header("Hyper Parameters")
    hyperparam_tune = st.selectbox(
        "Do you want to tune LLM?",
        ["No", "Yes"],
    )
    model = st.selectbox(
        "Select Model",
        ["llama3.1", "llama2"],
    )
    temperature = st.slider("Temperature", 0.01, 1.0, 0.5, 0.01)
    top_p = st.slider("Top_p", 0.01, 1.0, 0.9, 0.01)
    apply_changes = st.button("Apply Changes")


    # Store Query Parameters and Apply on Submit
    if "hyperparams" not in st.session_state:
        st.session_state.hyperparams = {
            "model": "llama3.1",
            "temperature": 0.5,
            "top_p": 0.9,
            "hyperparam_tune":"No"
        }

    if apply_changes:
        st.session_state.hyperparams = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "hyperparam_tune":hyperparam_tune
        }
        st.success("Hyper Parameters Updated!")










# Chat Interface
st.markdown("---") 
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Chat Input and Execution
if prompt := st.chat_input("Type your query here:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Fetching top 3 semantically similar docs..."):
            result = query_vector_db(
                API_URL,
                prompt,
                collection_name
            )
            if result["status"] == "success":
                st.write("Top 3 semantically similar docs: \n\n", result["details"])  # Display the answer
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["details"]}
                )
            else:
                st.error(f"Query failed: {result['details']}")


        st.markdown("<hr style='border: 1px dotted #bbb;'>", unsafe_allow_html=True)

        with st.spinner(f"Generating response using {st.session_state.hyperparams['model']}..."):
            hyperparams = st.session_state.hyperparams
            result = query_rag_llm(
                API_URL,
                prompt,
                collection_name,
                hyperparams,
            )
            if result["status"] == "success":
                st.write("Assistant: \n\n", result["details"])  # Display the answer
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["details"]}
                )
            else:
                st.error(f"Query failed: {result['details']}")