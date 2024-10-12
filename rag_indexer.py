from arxiv_embedding import embed_metadata, EMBEDDING_DIM
from pinecone_db import get_pinecone_client, get_or_create_index

def get_record(embedding, metadata):
    filtered_metadata_columns = ["id", "categories", "title"]
    metadata = {k: v for k, v in metadata.items() if k in filtered_metadata_columns}
    return {
        "id": f"{metadata['id']}#arxiv-metadata#scibert",
        "values": embedding,
        "metadata": metadata
    }

def build_arxiv_index():
    pc = get_pinecone_client()
    index_name = "arxiv-index"
    index = get_or_create_index(pc, index_name, EMBEDDING_DIM)
    metadatas, embeddings = embed_metadata("machine learning", 500)
    """
    eg:
     {
      "id": "A", 
      "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
      "metadata": {"genre": "comedy", "year": 2020}
    }
    """
    # zip into vectors list of embeddings and metadata
    vectors = map(get_record, embeddings, metadatas)
    index.upsert(
        vectors=vectors,
        namespace="arxiv-metadata",
    )
    print(f"Index {index_name} has been updated with {len(metadatas)} papers.")

if __name__ == "__main__":
    build_arxiv_index()