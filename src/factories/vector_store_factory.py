from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import StorageContext, VectorStoreIndex
from pinecone.exceptions import PineconeApiException
import os


def get_vector_store(store_type, **kwargs):
    """
    Factory to return a vector store instance based on tenant config.
    """
    if store_type == "pinecone":
        os.environ["PINECONE_API_KEY"] = kwargs.get("api_key")

        pc = Pinecone(api_key=kwargs.get("api_key"))

        try:
            pc.create_index(
                name=os.getenv("VECTOR_DB_INDEX_NAME"),
                dimension=1536,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region=os.getenv("VECTOR_DB_ENVIRONMENT")),
            )
            print(f"Created new Pinecone index: {os.getenv('VECTOR_DB_INDEX_NAME')}")
        except PineconeApiException as e:
            if e.status == 409:  # Index already exists
                print(f"Index '{os.getenv('VECTOR_DB_INDEX_NAME')}' already exists, using existing index")
            else:
                raise e

        pinecone_index = pc.Index(os.getenv("VECTOR_DB_INDEX_NAME"))

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        skip_indexing = kwargs.get("skip_indexing", False)

        if skip_indexing:
            print("Using existing index without re-indexing")
            # Connect to existing index without adding new documents
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=kwargs.get("embed_model"))
        else:
            print("Creating new index from documents")
            # Create index from documents (will add to existing or create new)
            index = VectorStoreIndex.from_documents(
                documents=kwargs.get("documents"),
                storage_context=storage_context,
                embed_model=kwargs.get("embed_model"),
            )

        return index

    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
