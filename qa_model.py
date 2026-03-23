import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import re

class PolicyQA:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def preprocess_context(self, context):
        """Clean and chunk context for better Q&A"""
        context = re.sub(r'\s+', ' ', context).strip()
        # Split into manageable chunks (max 512 tokens)
        sentences = re.split(r'(?<=[.!?])\s+', context)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < 450:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def answer_question(self, question, context, max_chunks=5):
        """Answer question using BERT Q&A"""
        chunks = self.preprocess_context(context)[:max_chunks]
        best_answer = {"score": 0, "answer": "", "context": ""}
        
        for chunk in chunks:
            result = self.qa_pipeline({
                "question": question,
                "context": chunk
            })
            
            if result["score"] > best_answer["score"]:
                best_answer = {
                    "score": result["score"],
                    "answer": result["answer"],
                    "context": chunk[:200] + "..." if len(chunk) > 200 else chunk
                }
        
        return best_answer
    
    def generate_questions(self, context, num_questions=5):
        """Generate relevant questions from context"""
        chunks = self.preprocess_context(context)
        questions = []
        
        question_templates = [
            "What is",
            "How does",
            "Who is responsible for",
            "When must",
            "Why is",
            "Where can"
        ]
        
        for i, chunk in enumerate(chunks[:num_questions]):
            template = question_templates[i % len(question_templates)]
            question = f"{template} {chunk[:50].strip()[:20]}?"
            questions.append(question)
        
        return questions
