from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def get_embedding_model(provider="openai", **kwargs):
    """
    Dynamically returns an embedding model based on the provider.
    """
    if provider == "azure":
        return AzureOpenAIEmbedding(
            model=kwargs.get("model"),
            api_key=kwargs.get("api_key"),
            azure_endpoint=kwargs.get("endpoint_url"),
            deployment_name=kwargs.get("deployment_name"),
            api_version=kwargs.get("api_version"),
        )

    return OpenAIEmbedding(
        model=kwargs.get("model", "text-embedding-3-small"),
        api_key=kwargs.get("api_key"),
    )
