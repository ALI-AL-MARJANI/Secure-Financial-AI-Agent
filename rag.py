import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
loader = TextLoader("data/bank_policies.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5}) 

@tool
def search_bank_policies(query: str) -> str:
    """
    Searches the official SecureBank internal policy documents.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant policy found in the knowledge base "
    context = "\n\n".join([doc.page_content for doc in results])
    return f"Retrieved Policy Information:\n{context}"