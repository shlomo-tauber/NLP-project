import semanticscholar

class SemanticScholarWrapper:
    def __init__(self):
        self.client = semanticscholar.SemanticScholar()

    def bulk_search(self, **kwargs):
        return self.client.search_paper(**kwargs, limit=100, bulk=True)