"""Day 3: Streamlit chat-style UI for the RAG assistant."""
import streamlit as st

from ask import answer

st.set_page_config(page_title="CiteRAG", page_icon="📚")
st.title("📚 CiteRAG")
st.caption("Réponses basées uniquement sur ta documentation, avec sources.")

question = st.text_input("Ta question")

if st.button("Demander") and question.strip():
    with st.spinner("Recherche en cours..."):
        text, hits = answer(question.strip())
    st.markdown(text)
    with st.expander("Passages retrouvés (à vérifier)"):
        for d, m, dist in hits:
            st.markdown(f"**{m['source']}**, p.{m['page']} (distance {dist:.3f})")
            st.text(d[:400])
