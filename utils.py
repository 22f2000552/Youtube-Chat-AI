import ssl
import re
import os
import requests
import yt_dlp

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

ssl._create_default_https_context = ssl._create_unverified_context

def get_transcript(url):

    settings = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "skip_download": True,
        "quiet": True
    }

    with yt_dlp.YoutubeDL(settings) as ydl:

        info = ydl.extract_info(url, download=False)

        subtitles = info.get("subtitles") or info.get("automatic_captions")

        if not subtitles or "en" not in subtitles:
            raise ValueError("❌ No English subtitles found for this video")

        en_subs = subtitles["en"]
        text_format = next((s for s in en_subs if s["ext"] == "vtt"), None)

        if not text_format:
            raise ValueError("❌ Could not find subtitle text")

        headers = {"User-Agent": "Mozilla/5.0"}
        raw = requests.get(text_format["url"], headers=headers).text

        lines = raw.split("\n")
        clean_lines = []
        for line in lines:
            line = line.strip()
            if "-->" in line: continue
            if line == "": continue
            if line == "WEBVTT": continue
            if re.match(r"^\d+$", line): continue
            clean_lines.append(line)

        seen = []
        for line in clean_lines:
            if line not in seen:
                seen.append(line)

        return " ".join(seen)

def build_vector_store(full_text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([full_text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

def ask_ai(question, vector_store):

    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    prompt = f"""
You are a helpful assistant. Use the following context to answer the question.

Transcript:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content