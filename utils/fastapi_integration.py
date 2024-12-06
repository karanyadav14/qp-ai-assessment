import os
from fastapi import FastAPI, UploadFile, File

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama


from utils.setup_milvus import (
    create_milvus_retriever
)

from utils.milvus_doc_insert import load_uploaded_documents, temp_write_uploaded_file


app = FastAPI()



@app.post("/upload/")
async def fastapi_upload_files(collection_name, file: UploadFile = File(...)):
    """
    Endpoint to upload a file and add its content to the vector database.
    """
    print(dir(file)) 
    # Save uploaded file temporarily
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = f"{temp_dir}/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    load_uploaded_documents(
    file_path,
    collection_name
    )
    return file_path






@app.post("/query/")
async def query_documents(query, collection_name):

    vectorstore = create_milvus_retriever(collection_name)
    try:
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)

        prompt_template = """
                Instructions: You are humble and polite AI assistant. Given following context in context tag respond to question.
                The response should be specific and concise. Keep answers around context provided.
                If question is outside of context provided or if you don't know the answer, just say that you don't know in a single line.
                Don't try to make up an answer.
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
        memory.save_context({"input":query},{"output":res})
        # memory.add_user_message(query)
        # memory.add_ai_message(res)

        return res
    except Exception as e:
        # st.error(f"Search failed: {str(e)}")
        return f"Search failed: {str(e)}"





