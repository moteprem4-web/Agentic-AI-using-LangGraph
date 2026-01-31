import streamlit as st
from new_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return str(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# ************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()


if "thread_names" not in st.session_state:
    st.session_state.thread_names = {}  # thread_id -> chat name

add_thread(st.session_state['thread_id'])

#*************** Helpers ************************
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

# **************************************** Sidebar UI *********************************
st.title("LangGraph Chatbot")
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

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



# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

     # first add the message to message_history
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})