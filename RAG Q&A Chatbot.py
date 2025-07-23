from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Step 1: Load and chunk documents
loader = TextLoader("data/my_docs.txt")  # or PDFLoader, etc.
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Step 2: Embeddings & Indexing
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Step 3: Load LLM (from HuggingFaceHub or OpenAI if free credits)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)

# Step 4: RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Step 5: Ask a question
query = "What is retrieval augmented generation?"
result = qa_chain({"query": query})

print("Answer:", result["result"])


# StreamLit UI --->>>>

import streamlit as st

st.title("📚 RAG Q&A Chatbot")

query = st.text_input("Ask something about your documents:")

if query:
    result = qa_chain({"query": query})
    st.write("### Answer:")
    st.write(result["result"])
