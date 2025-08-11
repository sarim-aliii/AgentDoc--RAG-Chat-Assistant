import streamlit as st
import uuid
import time
from core import (
    DatabaseChatMemory,
    load_forbidden_keywords,
    is_query_safe,
    setup_conversational_rag_chain,
    initialize_vector_store,  # <-- NEW IMPORT
    RELEVANCE_SCORE_THRESHOLD,
    DB_PATH,
    FORBIDDEN_KEYWORDS_PATH
)

# --- Page and Component Setup ---
st.set_page_config(page_title="AgentDoc", page_icon="🤖", layout="wide")
st.title("🤖 AgentDoc: Publication QA Assistant")

# --- NEW: Initialize Vector Store on App Start ---
# This will run once and be cached by Streamlit.
@st.cache_resource(show_spinner="Initializing knowledge base...")
def init_db():
    initialize_vector_store()

init_db()

@st.cache_resource(show_spinner="Initializing agent...")
def load_components():
    return setup_conversational_rag_chain()

rag_chain, vector_store = load_components()
forbidden_keywords = load_forbidden_keywords(FORBIDDEN_KEYWORDS_PATH)

# --- Session and DB Initialization (rest of the file is unchanged) ---
user_id = "web_user"
db_memory = DatabaseChatMemory(db_path=DB_PATH, user_id=user_id)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
    st.session_state.editing_session_id = None

# ... (The rest of your app.py file is exactly the same as your last version) ...
def start_new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
    st.session_state.editing_session_id = None

def switch_chat(session_id):
    st.session_state.session_id = session_id
    history = db_memory.get_session_history(session_id)
    st.session_state.messages = [{"role": msg.type, "content": msg.content} for msg in history]
    st.session_state.editing_session_id = None

def stream_simulator(text: str):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

with st.sidebar:
    st.header("Chat History")
    if st.button("➕ New Chat", use_container_width=True): start_new_chat()
    sessions = db_memory.get_all_sessions()
    for session_id, title in sessions:
        col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
        with col1:
            if st.button(title, key=f"select_{session_id}", use_container_width=True): switch_chat(session_id)
        with col2:
            if st.button("✏️", key=f"edit_{session_id}"): st.session_state.editing_session_id = session_id
        with col3:
            if st.button("🗑️", key=f"delete_{session_id}"):
                db_memory.delete_session(session_id)
                if st.session_state.session_id == session_id: start_new_chat()
                st.rerun()
        if st.session_state.get("editing_session_id") == session_id:
            with st.form(key=f"form_{session_id}"):
                new_title = st.text_input("New chat name", value=title)
                if st.form_submit_button("Save"):
                    db_memory.rename_session(session_id, new_title)
                    st.session_state.editing_session_id = None
                    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    db_memory.save_message(st.session_state.session_id, role="user", content=prompt)

    with st.chat_message("assistant"):
        response = ""
        if not is_query_safe(prompt, forbidden_keywords):
            response = "I am a document-based assistant and cannot respond to such requests."
            st.write(response)
        else:
            with st.spinner("Checking relevance..."):
                retrieved_docs = vector_store.similarity_search_with_score(prompt, k=1)
            
            if not retrieved_docs or retrieved_docs[0][1] > RELEVANCE_SCORE_THRESHOLD:
                response = "I'm sorry, but my knowledge is limited to the provided publication documents."
                st.write(response)
            else:
                with st.spinner("Thinking..."):
                    chat_history = db_memory.get_session_history(st.session_state.session_id)
                    chain_input = {"input": prompt, "chat_history": chat_history}
                    full_response = rag_chain.invoke(chain_input)
                    answer_text = full_response.get("answer", "I encountered an issue.")
                
                st.write_stream(stream_simulator(answer_text))
                response = answer_text

    st.session_state.messages.append({"role": "assistant", "content": response})
    db_memory.save_message(st.session_state.session_id, role="assistant", content=response)