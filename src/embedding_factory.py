from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
import os


def get_embedding_model(provider="azure"):
    if provider == "azure":
        return AzureOpenAIEmbedding(
            model=os.getenv("EMBEDDING_MODEL"),
            azure_endpoint=os.getenv("ENDPOINT_URL"),
            api_key=os.getenv("API_KEY"),
            deployment_name=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
            api_version=os.getenv("API_VERSION"),
        )
    elif provider == "openai":
        return OpenAIEmbedding(model=os.getenv("EMBEDDING_MODEL"))
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
