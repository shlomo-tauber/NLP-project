import semanticscholar

class SemanticScholarWrapper:
    def __init__(self):
        self.client = semanticscholar.SemanticScholar()

    def bulk_search(self, **kwargs):
        return self.client.search_paper(**kwargs, limit=100, bulk=True)
    
    def get_specter_embeddings(self, paper_ids):
        papers = self.client.get_papers(paper_ids, fields=["embedding.specter_v2"])
        def get_embedding_vector(p):
            if not p.embedding:
                return None
            return p.embedding["vector"]
        return map(get_embedding_vector, papers)