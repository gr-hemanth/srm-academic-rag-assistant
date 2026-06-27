import os
import base64
from collections import defaultdict

import streamlit as st
from PIL import Image

from build_index import build_index
from rag import get_answer, load_knowledge_base
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "SRMIST_logo.png"

def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

st.set_page_config(
    page_title="SRM Academic Assistant",
    page_icon=str(LOGO_PATH),
    layout="centered"
)

# ── Custom CSS for Claude-like chat layout ──────────────────────────────────
st.markdown(
    """
    <style>
        /* Hide all default avatar images */
        [data-testid="chatAvatarIcon-user"],
        [data-testid="chatAvatarIcon-assistant"] {
            display: none !important;
        }

        /* Remove left padding that was reserved for the avatar */
        [data-testid="stChatMessage"] {
            padding-left: 0 !important;
            gap: 0 !important;
        }

        /* ── User bubble: right-aligned ── */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
        [data-testid="stChatMessage"][class*="user"] {
            flex-direction: row-reverse !important;
            justify-content: flex-start !important;
        }

        /* Fallback selector for user messages */
        div[data-testid="stChatMessageContent"]:has(+ [data-testid="chatAvatarIcon-user"]),
        [data-testid="stChatMessage"]:nth-child(odd) {
            align-items: flex-end !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Custom chat renderer (bypasses st.chat_message avatars entirely) ─────────
def render_user_message(content: str):
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: flex-end;
            margin: 8px 0;
        ">
            <div style="
                background: #2e2e2e;
                color: #f0f0f0;
                padding: 10px 16px;
                border-radius: 18px 18px 4px 18px;
                max-width: 70%;
                font-size: 15px;
                line-height: 1.5;
                word-wrap: break-word;
            ">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_assistant_message(content: str):
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: flex-start;
            margin: 8px 0;
        ">
            <div style="
                background: transparent;
                color: #f0f0f0;
                padding: 10px 16px;
                border-radius: 18px 18px 18px 4px;
                max-width: 80%;
                font-size: 15px;
                line-height: 1.6;
                word-wrap: break-word;
            ">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
logo_base64 = get_base64(LOGO_PATH)
st.markdown(
    f"""
    <div style="
        display:flex;
        align-items:center;
        justify-content:center;
        gap:18px;
        margin-top:10px;
        margin-bottom:25px;
    ">
        <img src="data:image/png;base64,{logo_base64}" width="75">
        <h1 style="margin:0;font-size:42px;font-weight:700;">
            SRM Academic Assistant
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("SRM Academic Assistant")
st.sidebar.subheader("Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    if st.sidebar.button("Build Index", use_container_width=True):
        with st.spinner("Building index..."):
            build_index()
            load_knowledge_base()
        st.sidebar.success("Index rebuilt and reloaded.")

st.sidebar.subheader("Loaded Documents")

if os.path.exists("data"):
    pdfs = sorted([f for f in os.listdir("data") if f.endswith(".pdf")])
    if pdfs:
        for pdf in pdfs:
            st.sidebar.write(pdf)
    else:
        st.sidebar.caption("No PDFs available.")

if st.sidebar.button("Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ───────────────────────────────────────────────────────
for message in st.session_state.messages:
    if message["role"] == "user":
        render_user_message(message["content"])
    else:
        render_assistant_message(message["answer"])

        if message["sources"]:
            with st.expander("View Sources"):
                grouped = defaultdict(set)
                for pdf, page in message["sources"]:
                    grouped[pdf].add(page)
                for i, (pdf, pages) in enumerate(sorted(grouped.items()), start=1):
                    pages_str = ", ".join(str(p) for p in sorted(pages))
                    st.markdown(f"**{i}. {pdf}**\n\nPages: {pages_str}")

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input(
    "Ask a question about SRM regulations or hostel rules..."
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render_user_message(question)

    with st.spinner("Searching documents..."):
        answer, sources, _ = get_answer(question)

    render_assistant_message(answer)

    if sources:
        with st.expander("View Sources"):
            grouped = defaultdict(set)
            for pdf, page in sources:
                grouped[pdf].add(page)
            for i, (pdf, pages) in enumerate(sorted(grouped.items()), start=1):
                pages_str = ", ".join(str(p) for p in sorted(pages))
                st.markdown(f"**{i}. {pdf}**\n\nPages: {pages_str}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "answer": answer,
            "sources": sources,
        }
    )