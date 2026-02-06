import os
from httpcore import stream
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage,HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import sqlite3

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

llm = get_llm()

# LangGraph state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

#creating the database for invoke it 


conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)


# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)





