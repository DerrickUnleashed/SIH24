import pandas as pd
import streamlit as st
import os

# Load the CSV file (you can replace this with your file path)
file_path = 'DOLA_List_of_Advocate_Tamilnadu_2013_to_june2014.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')  # or use 'latin1'

st.set_page_config(page_title="Breaking Bonds", layout="centered")
col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("img.jpeg", use_column_width=True)

# Function to recommend lawyers based on input
def recommend_lawyers(df, place, court_type):
    recommendations = df[(df['ADDRESS'].str.contains(place, case=False, na=False)) & 
                         (df['SERVICE'].str.contains(court_type, case=False, na=False))]
    return recommendations[['ADVOCATE NAME', 'ADDRESS', 'SERVICE']]

# Streamlit interface
st.title("Lawyer Recommendation System")

# User input using Streamlit components
place_input = st.text_input("Enter the place:").upper()
court_type_input = st.selectbox("Select court type:", ["State", "Central"]).upper()

# Button to get recommendations
if st.button("Find Lawyers"):
    if place_input and court_type_input:
        recommended_lawyers = recommend_lawyers(df, place_input, court_type_input)
        
        if not recommended_lawyers.empty:
            st.write(f"Lawyers available in {place_input} for {court_type_input} court:")
            st.write(recommended_lawyers)
        else:
            st.write(f"No lawyers found in {place_input} for {court_type_input} court.")
    else:
        st.write("Please provide both place and court type.")

if st.button("Back to Homepage..."):
    st.write("Redirecting...")
    os.system("streamlit run app5.py")