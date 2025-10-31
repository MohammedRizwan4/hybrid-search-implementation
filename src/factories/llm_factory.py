from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI


def get_llm_model(provider="openai", **kwargs):
    """
    Dynamically returns an LLM model based on the provider.
    """
    if provider == "azure":
        return AzureOpenAI(
            model=kwargs.get("model"),
            api_key=kwargs.get("api_key"),
            azure_endpoint=kwargs.get("endpoint_url"),
            deployment_name=kwargs.get("deployment_name"),
            api_version=kwargs.get("api_version"),
        )

    return OpenAI(
        model=kwargs.get("model", "gpt-4o-mini"),
        api_key=kwargs.get("api_key"),
    )
