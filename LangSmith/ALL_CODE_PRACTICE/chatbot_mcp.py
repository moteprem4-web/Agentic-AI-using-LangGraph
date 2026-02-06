# ================== IMPORTS ==================
import os
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from mcp_adapter.client import MultiServerMCPClient

# ================== ENV ==================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ================== LLM ==================
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

# ================== MCP CLIENT ==================
client = MultiServerMCPClient(
    {
        "arith": {
            "command": [
                "python",
                "C:\\Users\\motep\\Desktop\\Chatbot\\arith_server.py"
            ]
        }
    }
)

# ================== STATE ==================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ================== CHAT NODE ==================
async def chat_node(state: ChatState):
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# ================== GRAPH ==================
async def build_graph():
    async with client:
        tools = await client.list_tools()
        print("MCP tools:", tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    return graph.compile()

# ================== RUN ==================
async def main():
    chatbot = await build_graph()

    result = await chatbot.ainvoke(
        {"messages": [HumanMessage(content="Hello MCP + LangGraph")]}
    )

    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
