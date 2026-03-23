import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

class PolicySummarizer:
    def __init__(self, model_name="t5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
    
    def clean_text(self, text):
        """Clean and preprocess policy text"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\$\$]', '', text)
        return text
    
    def summarize(self, text, max_length=512, min_length=50, num_beams=4):
        """Generate concise summary using T5"""
        cleaned_text = self.clean_text(text)
        
        # Truncate text if too long
        if len(cleaned_text) > 4000:
            cleaned_text = cleaned_text[:4000] + "..."
        
        input_text = f"summarize: {cleaned_text}"
        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                length_penalty=1.0
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.replace("summarize: ", "").strip()
    
    def generate_key_points(self, text, max_length=256):
        """Generate bullet-point style key points"""
        cleaned_text = self.clean_text(text[:2000])
        input_text = f"key points: {cleaned_text}"
        
        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_beams=3,
                do_sample=True,
                temperature=0.7
            )
        
        keypoints = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return keypoints.replace("key points: ", "").strip()
