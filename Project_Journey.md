***AgentDoc***
This document chronicles the development journey of AgentDoc, a Retrieval-Augmented Generation (RAG) powered chatbot. It details the evolution from a simple script to a robust, feature-rich web application, highlighting the key challenges faced and the engineering solutions implemented at each stage.


***Phase 1: Core RAG Pipeline and Initial Setup***
Objective: Build a simple command-line assistant that could answer questions from a custom JSON dataset.

**Steps Taken:**
1. Project Initialization: The project began with setting up a Python virtual environment and installing core dependencies: langchain, huggingface_hub, transformers, and chromadb.

2. Data Ingestion (ingest_data.py): 
    A script was created to load the publications.json data.
    The RecursiveCharacterTextSplitter was used to break down long publication descriptions into smaller, manageable chunks. This was identified as a crucial step for fitting context into the LLM's limited window.
    The HuggingFaceEmbeddings class was used with the sentence-transformers/all-MiniLM-L6-v2 model to convert these text chunks into semantic vectors.

3. Vector Storage: The generated embeddings and their corresponding text chunks were stored in a persistent Chroma vector database. This allowed us to separate the one-time ingestion process from the main application logic.

4. Initial Chatbot:
    A command-line interface was built using a simple while loop.
    The RetrievalQA chain was implemented, combining three components:
        1. Retriever: The ChromaDB store.
        2. LLM: The google/flan-t5-base model, chosen for its speed and suitability for question-answering.
        3. Prompt Template: A basic prompt instructing the LLM to answer questions based only on the provided   context.

**Challenges & Solutions:**
1. Challenge: The initial code had minor bugs, such as incorrect dictionary syntax (id: instead of 'id':).
Solution: Careful code review and debugging resolved these initial syntax issues.
2. Challenge: The first run produced a LangChainDeprecationWarning for many components.
Solution: We proactively updated the import paths (e.g., from langchain_community to langchain_chroma and langchain_huggingface) to align with the latest modular architecture of LangChain, ensuring future compatibility.


***Phase 2: Hardening the Application***
Objective: Move beyond a simple prototype and build a more robust, secure, and reliable system.

**Steps Taken:**
1. Prompt Injection Guardrails:
The initial defense was adding an instruction within the prompt itself to ignore malicious requests.
This was quickly proven insufficient. A stronger, Python-based is_query_safe function was implemented to check user input for a list of forbidden keywords before sending the query to the LLM chain.
To make this more manageable, the keywords were externalized into a forbidden_keywords.txt file.

2. Database Persistence:
Recognizing the need for audit trails and session management, a DatabaseChatMemory class was created.
Using Python's built-in sqlite3 library, a chat_history.db file was implemented to log every user query and assistant response with user IDs, session IDs, and timestamps.

3. Relevance and Hallucination Control:
When asked out-of-scope questions (e.g., "What is the capital of India?"), the bot was found to be "hallucinating" incorrect answers by trying to make sense of irrelevant retrieved documents.

*Solution 1* (Initial Attempt): A relevance score threshold was implemented using similarity_search_with_relevance_scores. This worked but produced a UserWarning because Chroma's distance metric was not normalized to the 0-1 scale that the function expected.

*Solution 2* (Robust Fix): We switched to similarity_search_with_score to get the raw distance metric. The logic was inverted to check if the score was below a certain threshold (since lower distance means higher relevance). This provided a reliable way to refuse to answer out-of-scope questions.

**Challenges & Solutions:**
1. Challenge: The stuff chain type caused Context Window Overflow errors, even with a small number of retrieved documents (k=2).
Solution: We upgraded the RAG chain type from "stuff" to "map_reduce", which processes documents individually, preventing overflow. This, however, introduced new validation errors because map_reduce doesn't accept a single custom prompt.

2. Challenge: The map_reduce chain itself was complex and still failed when a single document chunk was too large.
Solution (The Modern Approach): The entire RAG chain was refactored to use the modern LangChain Expression Language (LCEL). This provided a cleaner, more flexible, and more powerful way to build the chain that handled context and streaming elegantly. It also resolved the deprecation warnings associated with the older RetrievalQA class.


**Phase 3: Enhancing User Experience & Final Polish**
Objective: Transform the functional backend into a polished, user-friendly web application with advanced features.

**Steps Taken:**

1. Conversational Memory:
The system was failing on follow-up questions because it was stateless.
The RAG chain was upgraded to a create_history_aware_retriever and create_retrieval_chain. This new architecture first uses the chat history to rephrase a follow-up question into a standalone one before sending it to the retriever, enabling true conversational context.

2. Web Interface with Streamlit (app.py):
The core logic was refactored into a core.py file to separate it from the UI.
A new ui.py file was created using Streamlit to build the web interface.
Features included a real-time chat display, a "New Chat" button, and a persistent chat history sidebar loaded from the SQLite database.

3. Advanced UI Features:
Edit (✏️) and Delete (🗑️) buttons were added for each chat session in the sidebar, giving users full control over their history. This required a database schema upgrade to include a separate chat_sessions table.

4. Streaming Responses:
To improve perceived performance, the rag_chain.invoke() call was replaced with st.write_stream(rag_chain.stream()). This displays the LLM's response token-by-token, creating a live "typing" effect.

5. Upgrading the "Brain":
The LLM was upgraded from google/flan-t5-base to google/flan-t5-large for more nuanced and fluent answers.
A Re-ranker (BAAI/bge-reranker-base) was added to the retrieval pipeline. This two-stage process (fast retrieval followed by precise re-ranking) significantly improved the accuracy of the context provided to the LLM.


**Final Outcome**
The project successfully evolved from a simple CLI script into a full-featured, conversational AI assistant named AgentDoc. It boasts a robust RAG pipeline with conversational memory, a user-friendly web interface with persistent chat history, strong security guardrails, and a production-grade architecture that has been hardened through iterative debugging and enhancement. The journey from scratch was a practical lesson in the real-world challenges and engineering trade-offs involved in building reliable AI systems.