import requests
import streamlit as st

# --- Helper Functions ---
def upload_file_to_vector_db(api_url, file, collection_name):
    """Uploads a single file to the vector database."""
    try:
        payload = {"collection_name": collection_name}
        files = {"file": file}
        response = requests.post(f"{api_url}/upload/", params=payload, files=files)
        if response.status_code == 200:
            return {"status": "success", "details": response.json()}
        return {"status": "error", "details": response.text}
    except Exception as e:
        return {"status": "error", "details": str(e)}


def query_vector_db(api_url, prompt, collection_name):
    """Queries the vector database."""
    try:
        payload = {
            "query": prompt,
            "collection_name": collection_name
        }
        response = requests.post(f"{api_url}/semantic_similarity_search/", json=payload)
        
        if response.status_code == 200:
            return {"status": "success", "details": response.json()}
        return {"status": "error", "details": response.status_code}
    except Exception as e:
        return {"status": "error", "details": str(e)}


def query_rag_llm(api_url, prompt, collection_name, hyperparams):
    """Queries the rag based llm."""
    try:
        payload = {
            "query": prompt,
            "collection_name": collection_name,
            "hyperparams": hyperparams,
        }
        response = requests.post(f"{api_url}/query/", json=payload)
        if response.status_code == 200:
            return {"status": "success", "details": response.json()}
        return {"status": "error", "details": response.text}
    except Exception as e:
        return {"status": "error", "details": str(e)}
    



def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]





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


