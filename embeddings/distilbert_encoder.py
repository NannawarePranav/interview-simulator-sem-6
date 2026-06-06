from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class DistilBERTEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.cache = {}
        
    def encode(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract [CLS] token embedding
        cls_emb = outputs.last_hidden_state[0, 0, :].numpy()
        self.cache[text] = cls_emb
        return cls_emb
