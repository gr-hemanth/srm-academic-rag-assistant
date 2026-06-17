import os
import streamlit as st

from build_index import build_index
from rag import (
    get_answer,
    load_knowledge_base
)

st.set_page_config(
    page_title="SRM Academic Assistant",
    page_icon="🎓"
)

# =========================
# Sidebar
# =========================

st.sidebar.title("SRM Academic Assistant")
st.sidebar.subheader("Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if uploaded_file:

    save_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.sidebar.button("Build Index"):
        with st.spinner("Building index..."):
            build_index()
            load_knowledge_base()
        st.sidebar.success(
            "Index rebuilt and reloaded!"
    )
st.sidebar.subheader("Loaded Documents")

for file_name in sorted(os.listdir("data")):

    if file_name.endswith(".pdf"):

        st.sidebar.write(file_name)

if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []

    st.rerun()

# =========================
# Main Page
# =========================

st.title("🎓 SRM Academic Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Previous Messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input

question = st.chat_input(
    "Ask a question about SRM regulations or hostel rules..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching documents..."):
        answer, sources, _ = get_answer(question)

    source_text = "\n".join(
        f"- **{source_pdf}** — Page {page}"
        for source_pdf, page in sources
    )

    assistant_message = (
        f"{answer}\n\n"
        f"### Sources\n"
        f"{source_text}"
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_message
        }
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_message)