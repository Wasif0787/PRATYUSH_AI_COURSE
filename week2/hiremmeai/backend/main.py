import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from groq import Groq

from models import ChatRequest
from services import parse_resume, ask_candidate
from read_pdf import read_pdf

load_dotenv()

app = FastAPI()


resume = None


@app.on_event("startup")
def startup_event():
    global resume
    resume_text = read_pdf(Path(__file__).parent / "WasifHussain.pdf")
    print("Resume text extracted successfully")

    resume = parse_resume(resume_text)
    print("Resume parsed successfully")


@app.get("/")
def home():
    return {"message": " Welcome to the HireMeAI backend!"}


@app.post("/chat")
def chat(request: ChatRequest):
    if not resume:
        return {"error": "Resume is not parsed yet. Please try again later."}
    answer = ask_candidate(request.question, resume)
    return {"answer": answer}
