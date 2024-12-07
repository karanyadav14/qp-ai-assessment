### InsightPro: Retrieval Augmented Generation (RAG) based Contextual Chat Bot  

-  This chatbot allows you to upload multiple files and query them using the local LLaMA models.
  

##### Tools Used

- LLMs: LLaMA2, LLaMA3.1
- Vector DB: Milvus
- Embeddings: HuggingFaceEmbeddings
- Langchain, Langsmith
- Ollama
- FastAPI


##### RAG Architecture
![alt text](images/rag_architecture.png)


### Setup

#### Pre-requisite:
```
git clone git@github.com:karanyadav14/qp-ai-assessment.git
cd qp-ai-assessment
ENV_NAME=myenv && python3 -m venv $ENV_NAME && source $ENV_NAME/bin/activate && pip install ipykernel && python -m ipykernel install --user --name=$ENV_NAME
pip install -r requirements.txt
```

#### 1. Ollama
```
ollama pull llama2
ollama pull llama3.1
```

#### 2. Standalone Milvus DB
- Install standalone docker based milvus vector db and keep it running in background: 
```
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

- If asked for passward, enter system passward for admin access


#### 3. Launch FastAPI endpoint
```
uvicorn fastapi_integration:app --reload
```



### Launch Chatbot

- Limit size of file upload to 10 MB
```
streamlit run main.py --server.maxUploadSize 10
```



### Error Handling
- ERROR while file upload: Failed to upload <file>: HTTPConnectionPool(host='127.0.0.1', port=8000)
  - Ensure FastAPI server is up and running
