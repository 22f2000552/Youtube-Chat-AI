import streamlit as st
from utils import get_transcript, build_vector_store, ask_ai

st.set_page_config(
    page_title="YouTube Chat",
    page_icon="🎥"
)

st.title("🎥 Chat with YouTube Videos")

url = st.text_input("Paste a YouTube URL:")

if url:

    is_valid = "v=" in url or "youtu.be/" in url

    if not is_valid:
        st.error("❌ Please paste a valid YouTube URL")

    elif "vector_store" not in st.session_state or st.session_state.url != url:

        with st.spinner("⏳ Loading video..."):
            transcript = get_transcript(url)
            st.session_state.vector_store = build_vector_store(transcript)
            st.session_state.url = url
            st.session_state.messages = []

        st.success("✅ Ready! Ask anything about the video.")

if "vector_store" in st.session_state:

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask something about the video...")

    if user_input:

        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("🤔 Thinking..."):
            answer = ask_ai(user_input, st.session_state.vector_store)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })