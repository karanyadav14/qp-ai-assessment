import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from utils.setup_milvus import (
    create_milvus_collection,
    load_documents_to_milvus
)



    




def temp_write_uploaded_file(files):
    uploaded_files = []
    if files:
        st.write("Processing uploaded files...")
        for uploaded_file in files:
            bytes_data = uploaded_file.read()
            temp_dir = os.path.join(os.getcwd(), "../temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_file_path, "wb") as f:
                f.write(bytes_data)
            uploaded_files.append(temp_file_path)
        st.success("Processing Completed!!")

    return uploaded_files


def load_uploaded_documents(
    uploaded_files,
    collection_name,
     embedder_model="all-MiniLM-L6-v2"
):
    all_docs = []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file)[1].lower()

        # Select loader based on file type
        if file_extension == ".pdf":
            loader = PyPDFLoader(uploaded_file)
        elif file_extension == ".csv":
            loader = CSVLoader(uploaded_file)
        elif file_extension in [".txt", ".md"]:
            loader = TextLoader(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            continue

        documents = loader.load()
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=50)
            docs = splitter.split_documents(documents)
            all_docs.extend(docs)
        else:
            st.warning(f"No content found in file: {uploaded_file}")

    try:
        # Connect to Milvus and create the collection
        collection = create_milvus_collection(
            collection_name, dim=384
        )  

        # Insert documents into Milvus
        load_documents_to_milvus(collection, all_docs, embedder_model)
        return collection
    except:
        return "Failed to insert doc into vector db."