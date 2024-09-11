import streamlit as st
import os

st.set_page_config(page_title="Breaking Bonds", layout="centered")
col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("/Users/derricksamuel/Desktop/img.jpeg", use_column_width=True)

st.title("Homepage: ")

st.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

# Button for app2.py
if st.button("CHATBOT"):
    st.write("Redirecting...")
    os.system("streamlit run app2.py")
st.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

# Button for app4.py
if st.button("Lawyer Finder"):
    st.write("Redirecting...")
    os.system("streamlit run app4.py")