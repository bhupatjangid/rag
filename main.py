from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import fitz
import os

app = FastAPI()

retrieval_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

class Query(BaseModel):
    question: str

def generate_embeddings(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

def create_faiss_index(embeddings):
    d = embeddings.shape[1]  
    index = faiss.IndexFlatL2(d)  
    index.add(embeddings) 
    return index

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

def save_ncert_texts(file_path):
    all_texts = []
    text = extract_text_from_pdf(file_path)
    all_texts.append(text)
    return all_texts

ncert_texts = save_ncert_texts("iesc111.pdf")
ncert_embeddings = generate_embeddings(ncert_texts)
faiss_index = create_faiss_index(ncert_embeddings)

index = faiss_index
texts = ncert_texts 

def detect_intent(question):
    greeting_keywords = ["hello", "hi", "hey", "greetings"]
    if any(keyword in question.lower() for keyword in greeting_keywords):
        return "greeting"
    elif "summarize" in question.lower():
        return "summarize"
    else:
        return "retrieve"

@app.post("/query/")
async def query_rag_system(query: Query):
    intent = detect_intent(query.question)
    
    if intent == "greeting":
        return {"response": "Hello! How can I assist you today?"}
    
    elif intent == "summarize":
        long_text = " ".join(texts) 
        summary = summarization_pipeline(long_text, max_length=100, min_length=30, do_sample=False)
        return {"summary": summary[0]['summary_text']}

    else:
        question_embedding = retrieval_model.encode([query.question])

        D, I = index.search(np.array(question_embedding), k=5)
        retrieved_texts = [texts[i] for i in I[0]]

        context = " ".join(retrieved_texts)
        response = qa_pipeline(question=query.question, context=context)

        return {"question": query.question, "answer": response['answer']}

