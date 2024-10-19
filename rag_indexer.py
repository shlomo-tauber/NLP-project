#from arxiv_embedding import embed_metadata, EMBEDDING_DIM, embed_parallel
from pinecone_db import get_pinecone_client, get_or_create_index, METRIC_L2
from datetime import datetime
from semanticscholar_wrapper import SemanticScholarWrapper
import datasets
from specter_embedding import Specter2Document

def get_record(embedding, metadata):
    filtered_metadata_columns = ["id", "categories", "title"]
    metadata = {k: v for k, v in metadata.items() if k in filtered_metadata_columns}
    return {
        "id": f"{metadata['id']}#arxiv-metadata#scibert",
        "values": embedding,
        "metadata": metadata
    }

def build_arxiv_index():
    import arxiv_embedding
    pc = get_pinecone_client()
    # add date from now() formatted mm-dd-yyyy
    index_name = f"arxiv-index-{datetime.now().strftime('%m-%d-%Y')}"
    index = get_or_create_index(pc, index_name, arxiv_embedding.EMBEDDING_DIM)
    metadatas, embeddings = arxiv_embedding.embed_metadata("machine learning", 500, load_cache=False)
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

def build_semanticscholar_offline_dataset(limit=100):
    try:
        ds = datasets.load_from_disk("semanticscholar_dataset")
        return ds
    except FileNotFoundError:
        pass

    sch = SemanticScholarWrapper()
    def gen(shards):
        i = 0
        for year in shards:
            for x in sch.bulk_search(query="", fields_of_study=["Computer Science"], year=str(year)):
                if i > limit:
                    break
                i += 1
                yield dict(x)
                datasets.IterableDataset
    ds = datasets.Dataset.from_generator(gen, gen_kwargs={
        "shards": list(range(2018, 2019)),
    }, num_proc=4)
    ds.save_to_disk("semanticscholar_dataset")
    return ds


def get_semanticscholar_record(embedding_model_name):
    def map_record(row, embedding):
        filtered_metadata_columns = ["paperId", "title"]
        metadata = {k: v for k, v in row.items() if k in filtered_metadata_columns}
        return {
            "id": f"{row['paperId']}#semanticscholar-metadata#{embedding_model_name}",
            "values": embedding, #row['embeddings'],
            "metadata": metadata
        }
    return map_record

def semanticscholar_embedding_content(row):
    content = row['title']
    if 'abstract' in row and row['abstract']:
        content += " " + row['abstract']
    return content

def index_batch(embedding_model, index, b):
    ds = datasets.Dataset.from_dict(b)
    sch = SemanticScholarWrapper()    
    embeddings = sch.get_specter_embeddings(ds['paperId'])
    #embeddings = embedding_model.embed_parallel([semanticscholar_embedding_content(row) for row in ds])
    vectors = map(get_semanticscholar_record(embedding_model.embedding_model()), ds, embeddings)
    # skip missing embeddings
    filtered_vectors = filter(lambda x: x['values'] is not None, vectors)
    index.upsert(
        vectors=filtered_vectors,
        namespace="semanticscholar-metadata",
    )

def build_semanticscholar_index(embedding_model):
    pc = get_pinecone_client()
    index_name = f"semanticscholar-index-{embedding_model.embedding_model()}-{datetime.now().strftime('%m-%d-%Y')}"
    index = get_or_create_index(pc, index_name, embedding_model.embedding_dim(), metric=METRIC_L2)

    dataset = build_semanticscholar_offline_dataset()
    #dataset.with_format("torch")
    #dataset = dataset.map(semanticscholar_embedding_content)
    #dataset = dataset.map(lambda b: {'embeddings': embed_parallel(b['content']) }, batched=True, batch_size=250)
    dataset.map(lambda b: index_batch(embedding_model, index, b), batched=True, batch_size=500)

if __name__ == "__main__":
    #build_arxiv_index()
    embedding_model = Specter2Document()
    build_semanticscholar_index(embedding_model)