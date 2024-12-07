import os
from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama


from utils.setup_milvus import (
    create_milvus_retriever,
    search_milvus
)
from utils.milvus_doc_insert import load_uploaded_documents





app = FastAPI()

@app.post("/upload/")
async def fastapi_upload_files(collection_name: str, file: UploadFile=File(...)):
    """
    Endpoint to upload a file and add its content to the vector database.
    """

    # Save uploaded file temporarily
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = f"{temp_dir}/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    collection = load_uploaded_documents(
    [file_path],
    collection_name
    )
    return {"filename":file.filename, "collection":collection}




@app.post("/query/")
async def query_documents_using_llm(params:dict):

    collection_name = params["collection_name"]
    query = params["query"]
    hyperparams = params["hyperparams"]

    
    try:
        
        # Define prompt - future scope: pass through front end
        prompt_template = """
                Instructions: You are humble and polite AI assistant. 
                Given following context in context tag respond to question.
                The response should be specific and concise. Keep answers around context provided.
                If question is outside of context provided or if you don't know the answer, 
                just politely say that you don't know the answer in a single line.
                Don't try to make up an answer.
                Return answer in plain text instead of json.

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

        # Define ollama based llm model
        if hyperparams["hyperparam_tune"]=="Yes":
            llm = Ollama(
                model=hyperparams["model"],
                temperature=hyperparams["temperature"],
                top_p=hyperparams["top_p"]
                )
        else:
            llm = Ollama(
                model=hyperparams["model"]
                )
        
        # Top_k semantically similar (cosine similarity) chunks as a context
        retriever = create_milvus_retriever(
                            collection_name,
                            milvus_host="127.0.0.1",
                            milvus_port="19530",
                            embedder_model="all-MiniLM-L6-v2",
                            top_k = 3
                        )
        retrieved_context = retriever.retrieve_top_k(query, top_k=3)
        
        # Define chain
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Invoke the RAG chain with a question to get the response
        res = rag_chain.invoke({"context": retrieved_context, "question": query})

        
        # Future scope: memory integration
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        memory.save_context({"input":query},{"output":res})
        # memory.add_user_message(query)
        # memory.add_ai_message(res)

        return res
    
    except Exception as e:
        return f"Search failed: {str(e)}"




## Semantic similarity post request
@app.post("/semantic_similarity_search/")
async def semantic_similarity_search(params:dict):
    collection_name = params["collection_name"]
    query = params["query"]
    return search_milvus(collection_name, query)




# @app.exception_handler(500)
# async def internal_exception_handler(request: Request, exc: Exception):
#   return JSONResponse(status_code=500, content=jsonable_encoder({"code": 500, "msg": "Internal Server Error"}))