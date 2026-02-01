import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

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

model=get_llm()


prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)



parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in India'})

config={
    #for the name give to the runnable
    'run_name':'sequential-chain-demo',
    'tags':['llm app','report generation','summary generation'],
    'metadata':{'model_all':'meta-llama/Llama-3.2-3B-Instruct', 'project_name':'sequential-chain-demo'}
}

print(result)
