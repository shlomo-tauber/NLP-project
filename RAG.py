from arxiv_embedding import embed_text
import arxiv
from pylatexenc.latex2text import LatexNodes2Text
from transformers import AutoTokenizer
import transformers
import torch


def retrieve_relevant_papers(query, index, top_k=3):
    """
    The function retrive relevant title and citetion according the query.
    """
    # Generate the embedding for the query
    query_embedding = embed_text(query)

    # Search in the FAISS index for similar papers
    # Todo: add the search according to a model


def get_paper_citation_by_title(title, number_of_citations=10):
    # Search for the paper using its title
    res = []
    search = arxiv.Search(
        query=title,
        max_results=number_of_citations,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = list(search.results())

    if not results:
        return None, "No paper found with the given title."

    for paper in results:
        # Extract relevant information
        title = LatexNodes2Text().latex_to_text(paper.title)
        authors = ", ".join([author.name for author in paper.authors])
        year = paper.published.year
        arxiv_id = paper.get_short_id()
        url = paper.entry_id

        # Generate the citation
        citation = f"{authors}. ({year}). {title}. arXiv:{arxiv_id}. Retrieved from {url}"
        res.append((title, citation))
    return res


def generate_answer(query, titles, pipeline):
    """
    Generate an answer based on retrieved Arxiv papers.
    """

    context = ""
    # Append the title and citations of the papers to form a context
    for title in titles:
        citations = get_paper_citation_by_title(title)
        context += f"Title: {title}\n"
        for cite in citations:
            context += f"citations: {cite[0]}, {cite[1]}\n\n"

    # Prompt model to generate a response using the retrieved context
    input_prompt = f"What is the paper {query} citations, based on the next papers and citations: {context}\n"

    # # Generate a response
    sequences = pipeline(
        input_prompt,
        do_sample=True,
        top_k=50,
        top_p=0.7,
        num_return_sequences=1,
        repetition_penalty=1.1,
        max_new_tokens=500,
    )

    return sequences


def find_citetion_of_query(query):
    model = "PY007/TinyLlama-1.1B-Chat-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # create the indexes or use one allready exist
    index = None
    titles = retrieve_relevant_papers(query, index)
    sequences = generate_answer(query, titles, pipeline)
    filename = f"citetions of {query}"
    with open(filename, 'w') as file:
        file.write(sequences['generated_text'] + '\n')
