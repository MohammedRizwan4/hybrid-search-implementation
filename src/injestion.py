from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from embedding_factory import get_embedding_model
from vector_store_factory import get_vector_store
from dotenv import load_dotenv

load_dotenv()


def load_documents():
    return SimpleDirectoryReader("../playbook", recursive=True).load_data()


def build_index(store_type="pinecone", embedding_provider="azure"):
    docs = load_documents()
    embed_model = get_embedding_model(provider=embedding_provider)
    vector_store = get_vector_store(store_type=store_type)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
    print("✅ Documents indexed successfully!")


if __name__ == "__main__":
    build_index()
