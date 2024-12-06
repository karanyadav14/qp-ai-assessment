from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    Collection,
    DataType,
    utility,
)
from sentence_transformers import SentenceTransformer
import numpy as np

from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings("ignore")
from langsmith import traceable


def create_milvus_collection(collection_name, dim):
    connections.connect(alias="default", host="127.0.0.1", port="19530")
    utility.drop_collection(collection_name)

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            enable_dynamic_field=True,
            auto_id=True,
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(
            name="text", dtype=DataType.VARCHAR, max_length=65535
        )  ## Need proper chunking for inserting text
    ]

    schema = CollectionSchema(fields, description="Document embeddings")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection






def load_documents_to_milvus(collection, documents, embedder_model="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(embedder_model)
    texts = [doc.page_content for doc in documents]
    embeddings = embedder.encode(texts)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norm

    collection.insert(
        [normalized_embeddings, texts]
    )  ## Indexing will be handled by collection.create_index method
    collection.flush()
    collection.load()







def search_milvus(collection_name, query, embedder_model="all-MiniLM-L6-v2", top_k=5):
    embedder = SentenceTransformer(embedder_model)
    query_embedding = embedder.encode([query])

    norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    normalized_query_embeddings = query_embedding / norm

    connections.connect("default", host="127.0.0.1", port="19530")
    collection = Collection(collection_name)
    results = collection.search(
        data=normalized_query_embeddings,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["embedding", "text"],
    )

    result_texts = [(result.distance, result.entity.text) for result in results[0]]

    # Sort results by similarity score in descending order
    sorted_results = sorted(result_texts, key=lambda x: x[0], reverse=True)

    # Format the output: similarity score followed by the text
    formatted_results = "\n".join(
        [
            f"Cosine similarity score: {score:.4f} \nResponse: {text}\n\n"
            for score, text in sorted_results
        ]
    )

    return formatted_results






@traceable(run_type="retriever")
def create_milvus_retriever(
    collection_name,
    milvus_host="127.0.0.1",
    milvus_port="19530",
    embedder_model="all-MiniLM-L6-v2",
):

    try:

        embeddings = HuggingFaceEmbeddings(model_name=embedder_model)

        # Create Milvus retriever
        retriever = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"host": milvus_host, "port": milvus_port},
            vector_field="embedding"
        )

        return retriever
    except Exception as e:
        print(f"Error creating Milvus retriever: {e}")
        return None




