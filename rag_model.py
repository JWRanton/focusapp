from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class RAGModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("ehartford/Samantha-7B")
        self.model = AutoModelForCausalLM.from_pretrained("ehartford/Samantha-7B").to(self.device)

    def generate(self, query):
        prompt = f"You are an AI assistant specialized in helping people with ADHD manage tasks. Please provide a detailed step-by-step breakdown for the following task: {query}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
        
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response.split(prompt)[-1].strip()

# Initialize the model
rag_model = RAGModel()
