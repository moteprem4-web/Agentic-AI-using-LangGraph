import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -------------------------------------------------
# Load env vars
# -------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# -------------------------------------------------
# LLM (Hugging Face)
# -------------------------------------------------
@st.cache_resource
def get_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0.01,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
    )
    return ChatHuggingFace(llm=endpoint)

llm = get_llm()

# -------------------------------------------------
# PDF loading
# -------------------------------------------------
PDF_PATH = "islr.pdf"

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# -------------------------------------------------
# Chunking
# -------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
splits = splitter.split_documents(docs)

# -------------------------------------------------
# Hugging Face Embeddings (NO OpenAI)
# -------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -------------------------------------------------
# Prompt
# -------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY using the provided context. If the answer is not in the context, say 'I don't know'."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# -------------------------------------------------
# RAG Chain
# -------------------------------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# -------------------------------------------------
# Ask Question
# -------------------------------------------------
print("📄 PDF RAG ready (Hugging Face). Ctrl+C to exit.")

while True:
    q = input("\nQ: ")
    if not q.strip():
        break
    ans = chain.invoke(q)
    print("\nA:", ans)
