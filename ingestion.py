def create_index_if_not_exists(client, index_name):
    """
    Create an OpenSearch index with proper mapping for vector search if it doesn't exist.
    
    Args:
        client: OpenSearch client instance
        index_name: Name of the index to create
    """
    # Delete the index if it exists (to ensure proper mapping)
    if client.indices.exists(index=index_name):
        print(
            f"Deleting existing index '{index_name}' to recreate with proper mappings..."
        )
        client.indices.delete(index=index_name)


    # Define mappings with vector field for embeddings
    mappings = {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "content_type": {"type": "keyword"},
                "embedding": {"type": "knn_vector", "dimension":768},
                "filename": {"type": "keyword"},
            }
        },
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil",  # Use cosine similarity for embeddings
            }
        },
    }

    try:
        client.indices.create(index=index_name, body=mappings)
        print(f"Created index '{index_name}' with vector search capabilities.")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise
    
def prepare_chunks_for_ingestion(chunks):
  """
  Prepare chunks for ingestion by adding embeddings and token counts.

  Args:
      chunks: List of chunks to prepare

  Returns:
      List of prepared chunks ready for ingestion
  """
  from helper import get_embedding
  prepared_chunks = []
  for idx, chunk in enumerate(chunks):
    if not chunk.get("content"):
      print(f"skipping chunk {idx} due to missing content")
      continue
    chunk["embedding"] = get_embedding(chunk["content"])
    chunk_data = {
      "content": chunk["content"],
      "content_type": chunk.get("content_type", "text"),
      "embedding": chunk.get("embedding",None),
      "filename": chunk.get("filename", None),
    }
    prepared_chunks.append(chunk_data)
  return prepared_chunks


def ingest_chunks_into_opensearch(client, index_name, chunks):
  """
  Ingest prepared chunks into OpenSearch.

  Args:
      client: OpenSearch client instance
      index_name: Name of the index
      chunks: Prepared chunks with embeddings and token counts

  Returns:
      Number of successfully ingested documents
  """
  from opensearchpy import helpers
  actions = []
  for chunk in chunks:
    action = {
      "_index": index_name,
      "_source": chunk
    }
    actions.append(action)
  try:
    helpers.bulk(client, actions)
    print(f"ingested {len(chunks)} into index {index_name}")
  except Exception as e:
      print(f"error ingestion the chunk `{chunk}` into index {index_name}:{e}")
      raise
  
def ingest_all_content_into_opensearch(processed_images, processed_tables, semantic_chunks, index_name):
    """
    Process and ingest all content (images, tables, text) into OpenSearch.
    """

    from helper import get_opensearch_client

    # 1. Create OpenSearch client
    client = get_opensearch_client("localhost", 9200)

    # 2. Create index if it doesn't exist
    create_index_if_not_exists(client, index_name)

    # 3. prepare and ingest images
    image_chunks = prepare_chunks_for_ingestion(processed_images)
    ingest_chunks_into_opensearch(client,index_name, image_chunks)
    
    table_chunks = prepare_chunks_for_ingestion(processed_tables)
    ingest_chunks_into_opensearch(client,index_name, table_chunks)
    
    semantic_chunks = prepare_chunks_for_ingestion(semantic_chunks)
    ingest_chunks_into_opensearch(client,index_name, semantic_chunks)


if __name__ == "__main__":
  from unstructured.partition.pdf import partition_pdf
  from chunking import (
      create_semantic_chunks,
      process_images_with_captions,
      process_tables_with_description,
  )
  pdf_file_path = "files/paper.pdf"
  raw_chunks = partition_pdf(
    filename=pdf_file_path,
    strategy="hi_res",
    infer_table_structure=True,
    extract_image_block_types=["Image", "Figure", "Table"],
    extract_image_block_to_payload=True,
    chunking_strategy=None,
  )
  
  processed_images = process_images_with_captions(raw_chunks)
  
  processed_tables = process_tables_with_description(raw_chunks, use_gemini=True, use_ollama=False)
  
  text_chunks = partition_pdf(
    filename=pdf_file_path,
    strategy="hi_res",
    chunking_strategy="by_title",
    max_characters=2000,
    combine_text_under_n_chars=500,
    new_after_n_chars=1500
  )
  semantic_chunks = create_semantic_chunks(text_chunks)
  index_name = "localrag"
  ingest_all_content_into_opensearch(processed_images, processed_tables, semantic_chunks, index_name)
