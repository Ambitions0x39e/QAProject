class QueryResult():
    def __init__(self, desc, similarity):
        self.desc = desc
        self.similarity = similarity

    def debug_repr(self):
        return f"QueryResults<\ndesc={self.desc}\nsimilarity={self.similarity}\n>"
    
    def __repr__(self):
        return f"{self.desc}\n{self.similarity}"