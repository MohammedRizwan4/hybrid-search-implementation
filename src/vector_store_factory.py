from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os


def get_vector_store(store_type="pinecone"):
    if store_type == "pinecone":
        pinecone = Pinecone(api_key=os.getenv("VECTOR_DB_API_KEY"))
        index_name = os.getenv("VECTOR_DB_INDEX_NAME")

        existing_indexes = [idx["name"] for idx in pinecone.list_indexes()]

        if index_name not in existing_indexes:
            print(f"🆕 Creating new Pinecone index: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # must match your embedding size
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            print(f"✅ Using existing Pinecone index: {index_name}")

        # Connect to the index
        index = pinecone.Index(index_name)
        return PineconeVectorStore(pinecone_index=index)
    else:
        raise ValueError(f"Unsupported vector store: {store_type}")
