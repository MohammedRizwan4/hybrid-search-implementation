import os
from llama_index.core import VectorStoreIndex
from embedding_factory import get_embedding_model
from vector_store_factory import get_vector_store
from dotenv import load_dotenv
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import Settings
from llama_index.llms.azure_openai import AzureOpenAI

load_dotenv()

llm = AzureOpenAI(
    model=os.getenv("LLM_MODEL"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("API_KEY"),
    deployment_name=os.getenv("LLM_DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)


def query_documents(query: str, store_type="pinecone", embedding_provider="azure"):
    # Load vector store and embedding
    vector_store = get_vector_store(store_type)
    embed_model = get_embedding_model(provider=embedding_provider)

    Settings.llm = llm

    # Load existing index
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    vector_retriever = index.as_retriever(similarity_top_k=5)
    bm25_retriever = index.as_retriever(retriever_mode="bm25", similarity_top_k=5)

    # Create hybrid retriever (keyword + vector)
    fusion_retriever = QueryFusionRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        mode="reciprocal_rerank",
        use_async=True,
        llm=llm,
        similarity_top_k=5,
    )

    retrieved_nodes = fusion_retriever.retrieve(query)

    for node in retrieved_nodes:
        # print(f"\n📄 Document: {node.node_id}")
        # print(f"Score: {node.score}")
        # print(f"Text: {node.text}...")  # first 300 chars
        print(f"Metadata: {node.metadata}")


if __name__ == "__main__":
    query = input("Enter your query: ")
    result = query_documents(query)
    print("\n🔎 Top results:\n", result)
