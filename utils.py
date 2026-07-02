# ─── IMPORTS ─────────────────────────────────────────────────────────

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

# ─── SSL FIX (Mac) ───────────────────────────────────────────────────

ssl._create_default_https_context = ssl._create_unverified_context

# ─── FUNCTION 1 — GET TRANSCRIPT ─────────────────────────────────────

def get_transcript(url):

    settings = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "skip_download": True,
        "quiet": True
    }

    with yt_dlp.YoutubeDL(settings) as ydl:

        # Fetch video info
        info = ydl.extract_info(url, download=False)

        # Get subtitles
        subtitles = info.get("subtitles") or info.get("automatic_captions")

        if not subtitles or "en" not in subtitles:
            raise ValueError("❌ No English subtitles found for this video")

        # Find vtt format
        en_subs = subtitles["en"]
        text_format = next((s for s in en_subs if s["ext"] == "vtt"), None)

        if not text_format:
            raise ValueError("❌ Could not find subtitle text")

        # Download subtitles pretending to be a browser
        headers = {"User-Agent": "Mozilla/5.0"}
        raw = requests.get(text_format["url"], headers=headers).text

        # Clean up vtt format
        lines = raw.split("\n")
        clean_lines = []
        for line in lines:
            line = line.strip()
            if "-->" in line: continue
            if line == "": continue
            if line == "WEBVTT": continue
            if re.match(r"^\d+$", line): continue
            clean_lines.append(line)

        # Remove duplicates
        seen = []
        for line in clean_lines:
            if line not in seen:
                seen.append(line)

        return " ".join(seen)

# ─── FUNCTION 2 — BUILD VECTOR STORE ─────────────────────────────────

def build_vector_store(full_text):

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([full_text])

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store in FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

# ─── FUNCTION 3 — ASK AI ─────────────────────────────────────────────

def ask_ai(question, vector_store):

    # Find 3 most relevant chunks
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Connect to Groq
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    # Build prompt
    prompt = f"""
You are a helpful assistant. Use the following context to answer the question.

Transcript:
{context}

Question:
{question}
"""

    # Get answer
    response = llm.invoke(prompt)
    return response.content