from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Initial setup code (same as the Streamlit version but adapted)
embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Function to extract answers from the full response
def extract_answer(full_response):
    answer_start = full_response.find("Response:")
    if answer_start != -1:
        answer_start += len("Response:")
        return full_response[answer_start:].strip()
    return full_response

# Define the API endpoint for query processing
@app.post("/ask")
def ask_question(query: QueryRequest):
    input_prompt = query.query
    try:
        # Logic to process the query and get a response
        db_retriever = FAISS.load_local("ipc_embed_db")
        prompt_template = """
        You are a legal assistant. Use the following context to answer the user's query.

        {context}

        User's query: {question}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])
        
        # Using the Together API to handle the query
        llm = Together(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.5, max_tokens=1024, together_api_key="your_api_key_here")
        
        # Create a conversational retrieval chain
        qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=db_retriever, combine_docs_chain_kwargs={'prompt': prompt})
        
        # Get the result from the chain
        result = qa.invoke(input=input_prompt)
        
        # Extract the answer
        answer = extract_answer(result["answer"])
        
        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}
