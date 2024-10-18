from arxiv_embedding import embed_metadata, EMBEDDING_DIM, embed_parallel
from pinecone_db import get_pinecone_client, get_or_create_index
from datetime import datetime
from semanticscholar_wrapper import SemanticScholarWrapper
import datasets
import itertools

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
    # add date from now() formatted mm-dd-yyyy
    index_name = f"arxiv-index-{datetime.now().strftime('%m-%d-%Y')}"
    index = get_or_create_index(pc, index_name, EMBEDDING_DIM)
    metadatas, embeddings = embed_metadata("machine learning", 500, load_cache=False)
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

def build_semanticscholar_offline_dataset():
    try:
        ds = datasets.load_from_disk("semanticscholar_dataset")
        return ds
    except FileNotFoundError:
        pass

    sch = SemanticScholarWrapper()
    def gen():
        i = 0
        for x in sch.bulk_search(query="", fields_of_study=["Computer Science"], year="2018-"):
            if i > 3000:
                break
            i += 1
            yield dict(x)
    ds = datasets.Dataset.from_generator(gen)
    ds.save_to_disk("semanticscholar_dataset")
    return ds


def get_semanticscholar_record(row):
    filtered_metadata_columns = ["paperId", "title", "fieldsOfStudy"]
    metadata = {k: v for k, v in metadata.items() if k in filtered_metadata_columns}
    return {
        "id": f"{row['paperId']}#semanticscholar-metadata#scibert",
        "values": row['embeddings'],
        "metadata": metadata
    }

def semanticscholar_embedding_content(row):
    content = row['title']
    if 'abstract' in row and row['abstract']:
        content += " " + row['abstract']
    return { 'content': content }

def build_semanticscholar_index():
    dataset = build_semanticscholar_offline_dataset()
    dataset.with_format("torch")
    dataset = dataset.map(semanticscholar_embedding_content)
    dataset = dataset.map(lambda b: {'embeddings': embed_parallel(b['content']) }, batched=True, batch_size=50)
    pc = get_pinecone_client()
    index_name = f"semanticscholar-index-{datetime.now().strftime('%m-%d-%Y')}"
    index = get_or_create_index(pc, index_name, EMBEDDING_DIM)
    vectors = map(get_semanticscholar_record, dataset)
    index.upsert(
        vectors=vectors,
        namespace="semanticscholar-metadata",
    )
    print(f"Index {index_name} has been updated with {len(dataset)} papers.")

if __name__ == "__main__":
    #build_arxiv_index()
    build_semanticscholar_index()