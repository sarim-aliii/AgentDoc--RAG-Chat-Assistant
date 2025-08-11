***🤖 AgentDoc: A Conversational RAG Assistant***
AgentDoc is a sophisticated question-answering chatbot built with Python, LangChain, and Streamlit. It leverages the power of Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers from a custom set of documents.
This project demonstrates a complete, end-to-end workflow for building a modern AI assistant, from data ingestion and building a vector store to deploying a full-featured web application with conversational memory and a persistent chat history.

***Live Demo*** - (https://agentdoc--rag-chat-assistant.streamlit.app/)

✨ Features
Conversational Memory: Remembers previous turns in the conversation to understand follow-up questions.
Retrieval-Augmented Generation (RAG): Answers are grounded in a custom knowledge base (publications.json) to prevent hallucination.

Two-Stage Retrieval: Uses a fast vector search (ChromaDB) followed by a powerful cross-encoder re-ranker to ensure the most relevant context is used.

Interactive Web UI: A user-friendly, real-time chat interface built with Streamlit.
Persistent Chat History: Saves all conversations to a local SQLite database, allowing users to resume or review past chats.

Full Chat Management: Users can create new chats, rename session titles, and delete old conversations directly from the UI sidebar.

Streaming Responses: Assistant answers are streamed token-by-token for an engaging, real-time "typing" effect.
Robust Guardrails: Includes checks to prevent prompt injection attacks and gracefully handles out-of-scope questions.


🛠️ Tech Stack
Frameworks: LangChain, Streamlit
LLM: google/flan-t5-large (or any other Hugging Face model)
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
Re-ranker Model: BAAI/bge-reranker-base
Vector Store: ChromaDB
Database: SQLite


🚀 Getting Started
Follow these instructions to set up and run AgentDoc on your local machine.
1. Prerequisites
Python 3.9 or higher
An environment management tool like venv or conda (recommended)

2. Clone the Repository
git clone https://github.com/sarim-aliii/AgentDoc--RAG-Chat-Assistant.git
cd AgentDoc--RAG-Chat-Assistant

3. Set Up the Environment
Create and activate a virtual environment:
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

4. Install Dependencies
Install all the required Python packages from the requirements.txt file.
pip install -r requirements.txt

5. Ingest the Data
Before you can run the app, you need to build the vector store from the source documents. 
This is a one-time setup step.
python ingest_data.py
This will create a chroma_db directory in your project folder.

6. Run the Application
streamlit run ui.py

A new tab should automatically open in your web browser at http://localhost:8501. You can now start chatting with AgentDoc!

📂 Project Structure
AGENTDOC-RAG-CHAT-ASSISTANT/
│
├── 📄 .gitignore             # Files to ignore for Git
├── 📄 core.py                # Core backend logic (RAG chain, DB class)
├── 📄 forbidden_keywords.txt # List of keywords for security guardrails
├── 📄 ingest_data.py         # Script to build the vector store
├── 📄 publications.json      # The source knowledge base data
├── 📄 README.md              # This file
├── 📄 requirements.txt       # Python dependencies
└── 📄 app.py                  # The main Streamlit web application file

***Journey & Learnings***
This project evolved significantly from a simple CLI script. Key challenges and solutions are documented in the Project_Journey.md file, including:
Overcoming LLM context window limits by implementing advanced LangChain chains.
Building robust security and relevance guardrails.
Architecting a stateful application with a persistent database backend.
Refactoring code for a clean separation between the core logic and the user interface.