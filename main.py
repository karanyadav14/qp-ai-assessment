import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.llms import Ollama
from pymilvus import connections
from utils.setup_milvus import (
    create_milvus_collection,
    load_documents_to_milvus,
    search_milvus,
    create_milvus_retriever,
)
import os


import warnings
warnings.filterwarnings("ignore")






def load_uploaded_documents(
    uploaded_files,
    collection_name,
    embedder_model
):
    all_docs = []

    for uploaded_file in uploaded_files:
        try:
            file_extension = os.path.splitext(uploaded_file)[1].lower()

            # Select loader based on file type
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(temp_file_path)
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(temp_file_path)
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                continue

            documents = loader.load()
            if documents:
                splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=50)
                docs = splitter.split_documents(documents)
                all_docs.extend(docs)
            else:
                st.warning(f"No content found in file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading file {uploaded_file.name}: {str(e)}")

    # Connect to Milvus and create the collection
    collection = create_milvus_collection(
        collection_name, dim=384
    )  

    # Insert documents into Milvus
    load_documents_to_milvus(collection, all_docs, embedder_model)
    


def query_documents(query, uploaded_files, collection_name, embedder_model="all-MiniLM-L6-v2"):
    # Connect to Milvus
    try:
        connections.connect(alias="default", host="127.0.0.1", port="19530")
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {str(e)}")
        return "Error connecting to database."

    load_uploaded_documents(uploaded_files, collection_name, embedder_model)
    
    
    vectorstore = create_milvus_retriever(collection_name)
    try:
        
        prompt_template = """
                Instructions: You are an AI assistant and provides answers to questions by using fact based and contextual information provided.
                Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
                The response should be specific and concise. Keep answers around context provided.
                If you don't know the answer, just say that you don't know in a single line, don't try to make up an answer.
                <context>
                {context}
                </context>

                <question>
                {question}
                </question>

                Assistant:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables = ["context", "question"]
        )

        # Query the LLM
        llm = Ollama(model="llama2")
        retriever = vectorstore.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Invoke the RAG chain with a question to get the response
        res = rag_chain.invoke(query)
        return res
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return "Search error."



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

uploaded_files = []
if files:
    st.write("Processing uploaded files...")
    for uploaded_file in files:
        bytes_data = uploaded_file.read()
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(bytes_data)
        uploaded_files.append(temp_file_path)
    st.success("Processing Completed!!")

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
                collection_name = "multi_file_collection"

                prompt_with_memory = construct_prompt_with_memory(
                    st.session_state.messages, prompt
                )
                response = query_documents(prompt_with_memory, uploaded_files, collection_name)
                st.write(f"Answer: {response}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
    else:
        st.warning("No documents uploaded to query.")
