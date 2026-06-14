import streamlit as st

from rag import get_answer

st.set_page_config(
    page_title="SRM Academic Assistant",
    page_icon="🎓"
)

st.title("🎓 SRM Academic Assistant")

# =========================
# Chat History
# =========================

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

    # Show User Message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate Answer

    with st.spinner("Searching documents..."):
        answer, sources = get_answer(question)

    source_text = "\n".join(
        f"- **{source_pdf}** — Page {page}"
        for source_pdf, page in sources
    )

    assistant_message = (
        f"{answer}\n\n"
        f"### Sources\n"
        f"{source_text}"
    )

    # Save Assistant Message

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_message
        }
    )

    # Display Assistant Message

    with st.chat_message("assistant"):
        st.markdown(assistant_message)