import streamlit as st
import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

st.title("Chat with PDF using RAG")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    pdf = pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    extracted_text = ""

    for page in pdf:
        extracted_text += page.get_text()

    pdf.close()

    st.subheader("Extracted PDF Text")
    st.text_area("PDF Content", extracted_text, height=200)

    # Step 2: Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(extracted_text)

    st.subheader("Text Chunks")
    st.write("Total Chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        with st.expander(f"Chunk {i + 1}"):
            st.write(chunk)

    # Step 3: Generate Embeddings
    st.subheader("Generate Embeddings")

    if st.button("Generate Embeddings"):

        if not chunks:
            st.warning("No text chunks found.")
        else:
            try:
                embeddings_model = OllamaEmbeddings(
                    model="nomic-embed-text"
                )

                with st.spinner("Generating embeddings..."):
                    vectors = embeddings_model.embed_documents(chunks)

                st.success("Embeddings generated successfully!")

                st.write("Total Embeddings:", len(vectors))
                st.write("Vector Dimensions:", len(vectors[0]))

                st.write("First 10 values of first embedding:")
                st.write(vectors[0][:10])

            except Exception as e:
                st.error(f"Error generating embeddings: {e}")