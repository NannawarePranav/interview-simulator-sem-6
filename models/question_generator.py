from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GPT2QuestionGenerator:
    def __init__(self, model_path: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            print(f"[V2] Loaded GPT-2 from {model_path}")
        except Exception as e:
            print(f"[V2] Failed to load GPT-2 from {model_path}, falling back to base gpt2: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()

    def generate_followup(self, context: str, prev_answer: str, topic: str) -> str:
        prompt = f"Topic: {topic}\nContext: {context}\nCandidate Answer: {prev_answer}\nFollow-up Question:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        followup = generated_text[len(prompt):].strip()
        
        sentences = [s.strip() for s in followup.split('\n')[0].split('.') if s.strip()]
        if not sentences:
            return "Can you elaborate further on that?"
            
        first_sentence = sentences[0]
        if not first_sentence.endswith('?'):
            first_sentence += '?'
            
        return first_sentence
