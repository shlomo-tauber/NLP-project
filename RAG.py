#from arxiv_embedding import embed_text, EMBEDDING_DIM
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from pinecone_db import get_pinecone_client, get_or_create_index
from specter_embedding import Specter2AdhocQuery

class RAGRetriever:
    def __init__(self):
        self.pc = get_pinecone_client()
        self.index_name = "semanticscholar-index-specter2-10-19-2024"
        self.embedding_model = Specter2AdhocQuery()
        self.index = get_or_create_index(self.pc, self.index_name, self.embedding_model.embedding_dim())

    def retrieve_relevant_papers(self, query, top_k=3):
        query_embedding = self.embedding_model.embed_parallel(query).squeeze()
        vectors = self.index.query(vector=query_embedding.tolist(), namespace="semanticscholar-metadata", top_k=top_k, include_metadata=True)
        return vectors



PROMPT_TEMPLATE = (
    "You are a scholarly AI assistant helping to identify the source of a missing citation. A citation was deleted from an excerpt, "
    "indicated by '[CITATION]'. I will provide you with the excerpt, and you need to help me form a query to find the related papers.\n\n"
    "Rules:\n"
    "1. Use the context of the excerpt to form a single search query that captures the topic.\n"
    "2. I will return relevant papers for you to review, and you will then select the one most likely to be the source.\n"
    "3. If you cannot identify the source, respond with 'I don't know'.\n\n"
    "Excerpt:\n{excerpt}\n"
)
class CitationFinder:
    def __init__(self):
        model_name = "microsoft/Phi-3-mini-128k-instruct"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
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
        return self.pipeline(messages, **generation_args)[0]['generated_text']
        
    def find_citation(self, excerpt):
        messages = [PROMPT_TEMPLATE.format(excerpt=excerpt)]
        search_query = self.ask_model(*messages)
        messages.append(search_query)
        print(search_query)
        related_papers = self.retriever.retrieve_relevant_papers(search_query)
        print(related_papers)
        titles = [paper['metadata']['title'] for paper in related_papers['matches']]
        select_paper_message = '\n'.join([f"{i}. {title}" for (i, title) in enumerate(titles)])
        if not related_papers['matches']:
            select_paper_message = 'No papers found'
        messages.append(select_paper_message + '\nNow guess the cited paper title?')
        print(select_paper_message)
        citation = self.ask_model(*messages)
        print(citation)
        return citation, messages
