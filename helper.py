import requests
from opensearchpy import OpenSearch
def get_embedding(prompt, model="nomic-embed-text"):
    """
    Get the embedding for the given prompt using the specified model.

    Args:
        prompt (str): The prompt to embed.
        model (str): The model to use for embedding. Default is "nomic-embed-text".

    Returns:
        list: The embedding vector.
    """
    url = "http://localhost:11434/api/embeddings/"
    data = {
        "prompt": prompt,
        "model": model,
        "max_tokens":100,
        "stream":False,
        "temperature":0.7
        }
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json().get("embedding", None)


def get_opensearch_client(host, port):
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )

    if client.ping():
        print("Connected to OpenSearch!")
        info = client.info()
        print(f"Cluster name: {info['cluster_name']}")
        print(f"OpenSearch version: {info['version']['number']}")
    else:
        print("Connection failed!")
        raise ConnectionError("Failed to connect to OpenSearch.")
    return client

if __name__ == "__main__":
    get_opensearch_client("localhost", 9200)
