import streamlit as st

from rag import get_answer

st.set_page_config(
    page_title="SRM Academic Assistant",
    page_icon="🎓"
)

st.title("🎓 SRM Academic Assistant")

question = st.text_input(
    "Ask a question about SRM regulations or hostel rules:"
)

if st.button("Ask"):

    if question.strip():

        with st.spinner("Searching documents..."):
            answer, sources = get_answer(question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        for source_pdf, page in sources:
            st.markdown(
                f"- **{source_pdf}** — Page {page}"
            )