from arxiv_embedding import embed_text, EMBEDDING_DIM
import arxiv
from pylatexenc.latex2text import LatexNodes2Text
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from pinecone_db import get_pinecone_client, get_or_create_index

class RAGRetriever:
    def __init__(self):
        self.pc = get_pinecone_client()
        self.index_name = "arxiv-index"
        self.index = get_or_create_index(self.pc, self.index_name, EMBEDDING_DIM)

    def retrieve_relevant_papers(self, query, top_k=3):
        query_embedding = embed_text(query)
        vectors = self.index.query(query_embedding, top_k=top_k)
        return vectors
    
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

PROMPT_TEMPLATE = "You are given an excerpt from a paper, where a citation was deleted. I'm trying to find the citation (ignore the word [CITATION], that's just where the citation was deleted from. You will be asked to help me find the paper from which the citation was deleted.\n" \
    "You can ask me a search query and I will try to find related papers. You have to do it exactly once. After that, you will be asked to provide the citation. If you don't know the citation, you can say 'I don't know'.\n" \
    "Excerpt: {excerpt}\n"

class CitationFinder:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.retriever = RAGRetriever()

    def ask_model(self, *input):
        messages = [
            { "role": "system", "content": "You are a helpful AI assistant." },
        ]
        for i, message in enumerate(input):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({ "role": role, "content": message })
        
        generation_args = { 
            "max_new_tokens": 500, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 
        return self.pipeline(input, **generation_args)[0]['generated_text']
        
    def find_citation(self, excerpt):
        messages = [PROMPT_TEMPLATE.format(excerpt=excerpt)]
        search_query = self.ask_model(*messages)
        messages.append(search_query)
        related_papers = self.retriever.retrieve_relevant_papers(search_query)
        messages.append(related_papers)
        citation = self.ask_model(*messages)
        return citation, messages