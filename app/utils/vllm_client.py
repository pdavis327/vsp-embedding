import os
import httpx
from openai import OpenAI, APIError
from typing import List, Optional


def create_vllm_client(
    endpoint: str, model: str, api_key: Optional[str] = None
) -> Optional[OpenAI]:
    """
    Creates and configures an OpenAI client to connect to a vLLM endpoint.

    The vLLM endpoint must be compatible with the OpenAI API specification.

    Args:
        endpoint: The base URL of the vLLM server.
        model: The name of the model to use for embeddings.
        api_key: The API key, if required by the endpoint.

    Returns:
        An instance of the OpenAI client, or None if configuration is missing.
    """
    if not endpoint or not model:
        print("Error: The 'endpoint' and 'model' arguments are required.")
        return None

    # The OpenAI client's `base_url` should point to the versioned API path.
    # vLLM's OpenAI-compatible server typically serves from `/v1`.
    base_url = f"{endpoint.rstrip('/')}/v1"

    try:
        # --- Instantiate the OpenAI client ---
        # The `httpx.Client` is used to disable SSL certificate verification (`verify=False`),
        # which is often necessary for custom or private cloud deployments
        # that use self-signed certificates.
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
            or "not-needed",  # API key is required, but can be any string if not used
            http_client=httpx.Client(verify=False),
        )
        return client
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        return None


def get_embeddings(
    client: OpenAI, texts: List[str], model_name: str
) -> Optional[List[List[float]]]:
    """
    Generates embeddings for a list of texts using the provided OpenAI client.

    Args:
        client: An initialized OpenAI client instance.
        texts: A list of string inputs to embed.
        model_name: The name of the embedding model deployed on the server.

    Returns:
        A list of embedding vectors (each a list of floats), or None if an error occurs.
    """
    if not texts:
        print("Warning: Input text list is empty. Returning None.")
        return None

    try:
        # --- Call the embeddings API ---
        # The client library handles building the request and parsing the response.
        response = client.embeddings.create(
            model=model_name, input=texts, encoding_format="float"
        )

        # --- Extract the embedding vectors from the response object ---
        embeddings = [item.embedding for item in response.data]
        return embeddings

    except APIError as e:
        # Handle API-specific errors from the OpenAI library
        print(f"An API error occurred: {e}")
        print(f"  Status code: {e.status_code}")
        print(f"  Response: {e.response.text}")
        return None
    except Exception as e:
        # Handle other potential errors (e.g., network issues)
        print(f"An unexpected error occurred while getting embeddings: {e}")
        return None
