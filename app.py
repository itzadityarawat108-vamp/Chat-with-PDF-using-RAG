import streamlit as st
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("Chat with PDF using RAG")

# Step 1: Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    # Open PDF
    pdf = pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    # Extract text
    extracted_text = ""

    for page in pdf:
        extracted_text += page.get_text()

    pdf.close()

    # Display extracted text
    st.subheader("Extracted PDF Text")

    st.text_area(
        "PDF Content",
        extracted_text,
        height=250
    )

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
            st.write("Characters:", len(chunk))