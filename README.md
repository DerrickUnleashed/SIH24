# BreakingBonds: AI IPC Legal advice Assistant 📘

BharatLAW is a sophisticated legal advisory chatbot focused on providing detailed and contextually accurate responses about the Indian Penal Code. It utilizes a powerful combination of machine learning technologies to efficiently process and retrieve legal information.

---

## Features 🌟

- **Document Ingestion**: Automated processing of text documents to store legal information in a FAISS vector database.
- **Real-Time Interaction**: Real-time legal advice through a conversational interface built with Streamlit.
- **Legal Prompt Templating**: Structured prompt format ensuring clarity, detail, and legal accuracy in responses.


## Components 🛠️

### Ingestion Script (`Ingest.py`)

| Functionality        | Description |
|----------------------|-------------|
| **Document Loading** | Loads text documents from a specified directory. |
| **Text Splitting**   | Splits documents into manageable chunks for processing. |
| **Embedding Generation** | Utilizes `HuggingFace's InLegalBERT` to generate text embeddings. |
| **FAISS Database**   | Indexes embeddings for fast and efficient retrieval. |

### Web Application (`app.py`)

| Feature               | Description |
|-----------------------|-------------|
| **Streamlit Interface** | Provides a web interface for user interaction. |
| **Chat Functionality**  | Manages conversational flow and stores chat history. |
| **Legal Information Retrieval** | Leverages FAISS index to fetch pertinent legal information based on queries. 

---

## Setup 📦

### Prerequisites

- Python 3.8 or later
- ray
- langchain
- streamlit
- faiss

 
