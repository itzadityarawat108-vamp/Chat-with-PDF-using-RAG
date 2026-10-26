import streamlit as st
import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS

st.title("Chat with PDF using RAG")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:

    # Extract PDF text
    pdf = pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    extracted_text = ""

    for page in pdf:
        extracted_text += page.get_text() + "\n"

    pdf.close()

    st.success("PDF uploaded successfully!")

    st.subheader("Extracted Text")
    st.text_area("PDF Content", extracted_text, height=200)

    # Step 2: Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(extracted_text)

    st.write("Total Chunks:", len(chunks))

    # Step 3: Ollama Embedding Model
    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Step 4: Create FAISS Vector Store
    st.subheader("FAISS Vector Database")

    # Clear the old index when the uploaded file changes
    file_id = (
        uploaded_file.name,
        uploaded_file.size,
        uploaded_file.file_id
    )

    if st.session_state.get("indexed_file") != file_id:
        st.session_state.pop("vector_store", None)
        st.session_state.pop("indexed_file", None)

    if st.button("Create FAISS Index"):

        if not chunks:
            st.warning("No text found in PDF.")

        else:
            try:
                with st.spinner("Creating FAISS index..."):

                    vector_store = FAISS.from_texts(
                        texts=chunks,
                        embedding=embeddings_model
                    )

                    st.session_state.vector_store = vector_store
                    st.session_state.indexed_file = file_id

                st.success("FAISS index created successfully!")

            except Exception as e:
                st.error(f"Indexing error: {e}")

    # Step 5: Generate Answer using Ollama
    if "vector_store" in st.session_state:

        st.subheader("Ask Questions About Your PDF")

        question = st.text_input(
            "Enter your question"
        )

        if st.button("Generate Answer"):

            if not question.strip():
                st.warning("Please enter a question.")

            else:
                try:
                    with st.spinner("Generating answer..."):

                        # Retrieve relevant chunks
                        docs = st.session_state.vector_store.similarity_search(
                            question,
                            k=3
                        )

                        # Combine retrieved chunks
                        context = "\n\n".join(
                            doc.page_content for doc in docs
                        )

                        # Create prompt
                        prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the
provided PDF context.

If the answer is not present in the context,
say: "I could not find this information in the PDF."

PDF Context:
{context}

Question:
{question}

Answer:
"""

                        # Connect to Ollama LLM
                        llm = ChatOllama(
                            model="llama3.2",
                            temperature=0
                        )

                        # Generate answer
                        response = llm.invoke(prompt)

                    st.subheader("AI Answer")
                    st.write(response.content)

                    # Display retrieved sources
                    with st.expander("View Retrieved PDF Chunks"):
                        for i, doc in enumerate(docs):
                            st.write(f"**Chunk {i + 1}**")
                            st.write(doc.page_content)
                            st.divider()

                except Exception as e:
                    st.error(f"Error generating answer: {e}")