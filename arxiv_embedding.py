import arxiv
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# Load a pre-trained NLP model (SciBERT for scientific text embedding)
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def fetch_arxiv_metadata(query, max_results=500):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    metadata = []
    for result in search.results():
        metadata.append({
            'title': result.title,
            'abstract': result.summary,
        })
    return pd.DataFrame(metadata)


def embed_text(text):
    """
    Embed text (metadata) into vector space
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the CLS token embedding as a representation of the text
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings


def embed_metadata(query, max_results):
    """
    Embed arxiv metadata (title and abstract) into a vector space, and save them into pt file
    :param query: Subject of papers to embed
    :param max_results: Number of papers to embed
    """
    # Fetch metadata from the ArXiv API
    df = fetch_arxiv_metadata(query, max_results)

    # Embed the titles and abstracts into vector space
    embeddings = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Combine title and abstract to embed them together
        text_to_embed = row['title'] + " " + row['abstract']
        embedding = embed_text(text_to_embed)
        embeddings.append(embedding)

    # Convert embeddings list to a tensor
    all_embeddings = torch.cat(embeddings)

    # Save the embeddings to disk for future use
    torch.save(all_embeddings, f"arxiv_metadata_embeddings_{query}.pt")


if __name__ == '__main__':
    embed_metadata("hypergraphs", 500)
