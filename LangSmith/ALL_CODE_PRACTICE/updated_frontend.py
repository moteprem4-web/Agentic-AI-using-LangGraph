import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ============================ Utility Functions ============================

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    add_thread(thread_id)

    # empty chat
    st.session_state.message_history = []

    # placeholder name (important)
    st.session_state.thread_names[thread_id] = "New Conversation"

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


# ============================ Session Setup ============================

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

if "thread_names" not in st.session_state:
    st.session_state.thread_names = {}  # thread_id -> chat name

add_thread(st.session_state.thread_id)

# create name if missing
if st.session_state.thread_id not in st.session_state.thread_names:
    st.session_state.thread_names[st.session_state.thread_id] = "New Conversation"


# ============================ Helpers ============================

def get_or_create_thread_name(thread_id):
    if thread_id in st.session_state.thread_names:
        return st.session_state.thread_names[thread_id]

    messages = load_conversation(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            name = msg.content.strip()[:35]
            st.session_state.thread_names[thread_id] = name
            return name

    return "New Conversation"


# ============================ Sidebar UI ============================

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

# build labels
chat_labels = [
    get_or_create_thread_name(tid)
    for tid in st.session_state.chat_threads
]

label_to_id = dict(zip(chat_labels, st.session_state.chat_threads))

current_label = get_or_create_thread_name(st.session_state.thread_id)

selected_label = st.sidebar.radio(
    label="",
    options=chat_labels,
    index=chat_labels.index(current_label)
)

selected_thread_id = label_to_id[selected_label]
st.session_state.thread_id = selected_thread_id

# load selected chat
messages = load_conversation(selected_thread_id)
st.session_state.message_history = [
    {
        "role": "user" if isinstance(m, HumanMessage) else "assistant",
        "content": m.content
    }
    for m in messages
]


# ============================ Main Chat UI ============================

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    thread_id = st.session_state.thread_id

    # rename chat on first question
    if st.session_state.thread_names[thread_id] == "New Conversation":
        st.session_state.thread_names[thread_id] = user_input.strip()[:35]

    # user message
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {"configurable": {"thread_id": thread_id}}

    with st.chat_message("assistant"):

        def ai_only_stream():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessage):
                    yield chunk.content

        ai_message = st.write_stream(ai_only_stream())

    # assistant message
    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_message}
    )
