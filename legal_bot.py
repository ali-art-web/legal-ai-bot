import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class LegalBot:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.docs = []
        self.index = faiss.IndexFlatL2(384)
    
    def add_law(self, text, name="", article=""):
        vec = self.model.encode([text])
        self.index.add(vec)
        self.docs.append({"text": text, "name": name, "article": article})
    
    def ask(self, question):
        q_vec = self.model.encode([question])
        D, I = self.index.search(q_vec, 3)
        
        if len(I[0]) == 0:
            return "Не нашёл информации в базе."
        
        laws_found = []
        for idx in I[0]:
            if idx < len(self.docs):
                law = self.docs[idx]
                laws_found.append(f"{law['name']} {law['article']}: {law['text'][:100]}...")
        
        answer = "Найденные правовые нормы:\n"
        answer += "\n".join([f"• {law}" for law in laws_found])
        answer += "\n\nПроконсультируйтесь с юристом для применения."
        
        return answer
