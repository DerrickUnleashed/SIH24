import time
import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together
from langchain_huggingface import HuggingFaceEmbeddings
from footer import footer
import pvporcupine
from pvrecorder import PvRecorder
import pvleopard
import threading
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

load_dotenv()
apikey = os.getenv("API_KEY")
st.set_page_config(page_title="Breaking Bonds", layout="centered")
col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("/Users/derricksamuel/Desktop/img.jpeg", use_column_width=True)

def hide_hamburger_menu():
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

hide_hamburger_menu()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

directory_path = '/Users/derricksamuel/Desktop/BreakingBonds-Chatbot/data'
faiss_index_path = "ipc_embed_db/index.faiss"

if not os.path.isfile(faiss_index_path):
    try:
        loader = DirectoryLoader(directory_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(
            model_name="law-ai/InLegalBERT"
        )
        faiss_db = FAISS.from_documents(texts, embeddings)
        faiss_db.save_local("ipc_embed_db")
    except Exception as e:
        st.error(f"Error creating or saving FAISS index: {e}")
        st.stop()

embeddings = load_embeddings()
db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_template = """
<s>[INST]
As a legal chatbot specializing in the Indian Penal Code, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
- Respond in a bullet-point format to clearly delineate distinct aspects of the legal query.
- Each point should accurately reflect the breadth of the legal provision in question, avoiding over-specificity unless directly relevant to the user's query.
- Clarify the general applicability of the legal rules or sections mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
- Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
- Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations unless otherwise specified.
- Conclude with a brief summary that captures the essence of the legal discussion and corrects any common misinterpretations related to the topic.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
- [Detail the first key aspect of the law, ensuring it reflects general application]
- [Provide a concise explanation of how the law is typically interpreted or applied]
- [Correct a common misconception or clarify a frequently misunderstood aspect]
- [Detail any exceptions to the general rule, if applicable]
- [Include any additional relevant information that directly relates to the user's query]
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])
llm = Together(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.5, max_tokens=1024, together_api_key="2c08606d41a385db6efcb6ef28b81284cddf874e68fcd7e22c7fe4139a19b603")

qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=st.session_state.memory, retriever=db_retriever, combine_docs_chain_kwargs={'prompt': prompt})

def extract_answer(full_response):
    answer_start = full_response.find("Response:")
    if answer_start != -1:
        answer_start += len("Response:")
        answer_end = len(full_response)
        return full_response[answer_start:answer_end].strip()
    return full_response

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def recognize_speech_with_picovoice():
    freq = 44100
    duration = 10
    porcupine = pvporcupine.create(access_key=apikey,keywords=["jarvis"])
    
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
    
    recorder.start()
    st.write("Waiting for Passkey...")
    
    while True:
        pcm = recorder.read()
        
        # Process the audio frame by Porcupine
        keyword_index = porcupine.process(pcm)
        
        # If the keyword is detected, break the loop
        if keyword_index >= 0:
            st.write("Authorised!")
            st.write("Recording Started...")
            break
                
    st.write("Recording...for 10 secs")
    try:
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=1, dtype='int16')
        sd.wait()
        print("Recording complete.")
        write("/Users/derricksamuel/Desktop/BreakingBonds-Chatbot/temp.wav", freq, recording)
        print("Audio files saved.")
    except Exception as e:
        print(f"An error occurred: {e}")
    st.write("Processing the Voice...")
    leopard = pvleopard.Leopard(
                access_key=apikey,
                model_path="/Users/derricksamuel/Desktop/BreakingBonds-Chatbot/leopard_params.pv",
                library_path="/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/pvleopard/lib/mac/arm64/libpv_leopard.dylib"
            )
    transcript, words = leopard.process_file('/Users/derricksamuel/Desktop/BreakingBonds-Chatbot/temp.wav')
    print(transcript)
    leopard.delete()
    return transcript.strip()


input_prompt = st.chat_input("Enter your query...")

st.write("Or, use voice input:")
if st.button("Use Voice Input 🎤"):
    input_prompt = recognize_speech_with_picovoice()

if input_prompt:
    with st.chat_message("user"):
        st.markdown(f"**You:** {input_prompt}")

    st.session_state.messages.append({"role": "user", "content": input_prompt})
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            result = qa.invoke(input=input_prompt)
            message_placeholder = st.empty()
            answer = extract_answer(result["answer"])

            full_response = "\n\n"
            for chunk in answer:
                full_response += chunk
                time.sleep(0.002)
                message_placeholder.markdown(full_response + "\n\n_______________________________________________________________________\n", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.button('Start a New Chat 💬', on_click=reset_conversation):
            st.experimental_rerun()

if st.button("Back to Homepage..."):
    st.write("Redirecting...")
    os.system("streamlit run app5.py")

footer()