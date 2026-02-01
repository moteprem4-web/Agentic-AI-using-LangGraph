import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

#how to set new langsmith project name and also the project name in the .env file
#directtly from the code

import os

# os.environ["LANGCHAIN_PROJECT"] = "sequential-chain-demo" 


# LLM loader (cached)
@st.cache_resource
def get_llm():
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0.01,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
    )
    return ChatHuggingFace(llm=llm_endpoint)

# Prompt
prompt = PromptTemplate.from_template("{question}")

# Output parser
parser = StrOutputParser()

# Load model
model = get_llm()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run chain
result = chain.invoke({"question": "What is the capital of India?"})
print(result)
print(os.getenv("LANGCHAIN_TRACING_V2"))
print(os.getenv("LANGCHAIN_PROJECT"))


