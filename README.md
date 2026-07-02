# Youtube-Chat-AI

A Streamlit app that lets you chat with YouTube videos by extracting their subtitles, turning the transcript into searchable chunks, and answering questions with an AI model.

## Features

- Paste a YouTube link and load the video transcript
- Build a FAISS vector store from the transcript
- Ask questions about the video in a chat interface
- Uses Groq for the language model and Hugging Face embeddings for retrieval

## Requirements

- Python 3.10 or newer
- A valid Groq API key
- A YouTube video with English subtitles or automatic captions available

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Run the app

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

## How to use

1. Paste a valid YouTube URL into the input box.
2. Wait while the app loads the transcript and builds the vector store.
3. Ask questions about the video in the chat box.

## Notes

- The app only works when subtitles or automatic captions are available.
- The first load for a video can take a little longer because the transcript has to be fetched and embedded.
- If you change the YouTube link, the chat history for that video is reset.

## Project structure

- `app.py` - Streamlit UI and chat flow
- `utils.py` - Transcript loading, vector store creation, and AI answering logic

