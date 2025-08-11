__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
                                         

import sqlite3
import uuid
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder



EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "./chroma_db"
LLM_MODEL_ID = "google/flan-t5-large"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"
DB_PATH = "chat_history.db"
FORBIDDEN_KEYWORDS_PATH = "forbidden_keywords.txt"


class DatabaseChatMemory:
    def __init__(self, db_path, user_id):
        self.db_path = db_path
        self.user_id = user_id
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables_if_not_exists()

    def _create_tables_if_not_exists(self):
        cursor = self.conn.cursor()

        cursor.execute("CREATE TABLE IF NOT EXISTS chat_sessions (session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL);")

        cursor.execute("CREATE TABLE IF NOT EXISTS chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, timestamp TEXT NOT NULL, FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id));")
        self.conn.commit()

    def save_message(self, session_id: str, role: str, content: str):
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,))
        if cursor.fetchone() is None:
            title = content if role == 'user' else "New Chat"
            self.conn.execute("INSERT INTO chat_sessions (session_id, user_id, title, created_at) VALUES (?, ?, ?, ?)", (session_id, self.user_id, title, datetime.now().isoformat()))

        timestamp_str = datetime.now().isoformat()

        cursor.execute("INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (session_id, role, content, timestamp_str))

        self.conn.commit()

    def get_session_history(self, session_id: str):
        cursor = self.conn.cursor()

        cursor.execute("SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))

        return [HumanMessage(content=c) if r == 'user' else AIMessage(content=c) for r, c in cursor.fetchall()]
    
    def get_all_sessions(self):
        cursor = self.conn.cursor()

        cursor.execute("SELECT session_id, title FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC", (self.user_id,))
        
        return cursor.fetchall()
    
    def delete_session(self, session_id: str):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        self.conn.commit()

    def rename_session(self, session_id: str, new_title: str):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE chat_sessions SET title = ? WHERE session_id = ?", (new_title, session_id))
        self.conn.commit()

    def close(self):
        if self.conn: self.conn.close()


def load_forbidden_keywords(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        return []
    
def is_query_safe(query: str, forbidden_keywords: list[str]) -> bool:
    lower_query = query.lower()

    for keyword in forbidden_keywords:
        if keyword in lower_query:
            return False
        
    return True


def setup_conversational_rag_chain():
    embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME, 
                    model_kwargs={'device': 'cpu'}
                )
    vector_store = Chroma(
                    persist_directory=PERSIST_DIRECTORY, 
                    embedding_function=embeddings
                )
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    cross_encoder_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_ID)
    reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=3) 

    compression_retriever = ContextualCompressionRetriever(
                                base_compressor=reranker, 
                                base_retriever=base_retriever
                            )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_ID)
    pipe = pipeline(
                "text2text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_length=1024
            )
    llm = HuggingFacePipeline(pipeline=pipe)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", "Given a chat history and the latest user question... formulate a standalone question..."), 
        MessagesPlaceholder("chat_history"), ("human", "{input}")])

    history_aware_retriever = create_history_aware_retriever(llm, compression_retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([("system", "Answer the user's question based only on the following context:\n\n{context}"), 
        MessagesPlaceholder("chat_history"), ("human", "{input}")])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain