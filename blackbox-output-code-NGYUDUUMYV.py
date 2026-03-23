from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2
import docx
import io
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# Import your existing modules
from summarizer import PolicySummarizer
from qa_system import PolicyQA

app = FastAPI(title="Health Policy Simplifier API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
summarizer = None
qa_system = None

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 200

class QARequest(BaseModel):
    question: str
    context: str

@app.on_event("startup")
async def load_models():
    global summarizer, qa_system
    print("Loading AI models...")
    summarizer = PolicySummarizer()
    qa_system = PolicyQA()
    print("Models loaded successfully!")

@app.post("/api/summarize")
async def summarize_text(request: SummaryRequest):
    try:
        summary = summarizer.summarize(request.text, max_length=request.max_length)
        keypoints = summarizer.generate_key_points(request.text)
        
        return {
            "success": True,
            "summary": summary,
            "keypoints": keypoints,
            "word_count": len(request.text.split())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask_question(request: QARequest):
    try:
        answer = qa_system.answer_question(request.question, request.context)
        return {
            "success": True,
            **answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-questions")
async def generate_questions(request: SummaryRequest):
    try:
        questions = qa_system.generate_questions(request.text, 5)
        return {
            "success": True,
            "questions": questions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        if file.filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        elif file.filename.endswith('.docx'):
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])
        
        else:
            text = content.decode('utf-8')
        
        return {
            "success": True,
            "text": text[:10000],  # Limit for API
            "filename": file.filename,
            "word_count": len(text.split())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Health Policy Simplifier API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)