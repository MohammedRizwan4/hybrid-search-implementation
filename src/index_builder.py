from ingestion.injestion import load_documents

from factories.vector_store_factory import get_vector_store
from factories.embedding_factory import get_embedding_model
from factories.llm_factory import get_llm_model

from dotenv import load_dotenv
import os

load_dotenv()


def build_index(store_type="pinecone", skip_indexing=True):
    if not skip_indexing:
        documents = load_documents()
    else:
        documents = None
        print("Skipping document loading and indexing - using existing index")

    embed_model = get_embedding_model(
        provider="azure",
        api_key=os.getenv("API_KEY"),
        endpoint_url=os.getenv("ENDPOINT_URL"),
        deployment_name=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
        model=os.getenv("EMBEDDING_MODEL"),
        api_version=os.getenv("API_VERSION"),
    )

    llm_model = get_llm_model(
        provider="azure",
        api_key=os.getenv("API_KEY"),
        endpoint_url=os.getenv("ENDPOINT_URL"),
        deployment_name=os.getenv("LLM_DEPLOYMENT_NAME"),
        model=os.getenv("LLM_MODEL"),
        api_version=os.getenv("API_VERSION"),
    )

    index = get_vector_store(
        store_type=store_type,
        index_name=os.getenv("VECTOR_DB_INDEX_NAME"),
        environment="us-east-1",
        api_key=os.getenv("VECTOR_DB_API_KEY"),
        documents=documents,
        embed_model=embed_model,
        skip_indexing=skip_indexing,
    )

    query_engine = index.as_query_engine(
        vector_store_query_mode="hybrid",
        llm=llm_model,
    )

    response = query_engine.query("Peer Bonus Policy, Leave Policy, Code of Conduct")

    print(response)


if __name__ == "__main__":
    build_index("pinecone", skip_indexing=True)
