# ================== IMPORTS ==================
import os
import sqlite3
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import ChatHuggingFace

from langchain_core.messages import BaseMessage,HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# ================== ENV ==================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ================== LLM ==================
@st.cache_resource
def get_llm():
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0.01,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
         streaming=False
    )
    return ChatHuggingFace(llm=llm_endpoint)

llm = get_llm()

# ================== TOOLS ==================
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform add, sub, mul, div on two numbers."""
    try:
        if operation == "add":
            return {"result": first_num + second_num}
        if operation == "sub":
            return {"result": first_num - second_num}
        if operation == "mul":
            return {"result": first_num * second_num}
        if operation == "div":
            if second_num == 0:
                return {"error": "Division by zero"}
            return {"result": first_num / second_num}
        return {"error": "Unsupported operation"}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch stock price using Alpha Vantage."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=20XJKKLVOW74JTJZ"
    )
    return requests.get(url).json()

tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

# ================== STATE ==================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ================== NODES ==================
def chat_node(state: ChatState):
    """LLM Node that may anwser or request a tool call."""
    # Limit history to avoid HF crashes
    messages = state["messages"][-6:]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ================== CHECKPOINTER ==================
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# ================== GRAPH ==================
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# ================== THREAD UTILS ==================
def retrieve_all_threads():
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(threads)
