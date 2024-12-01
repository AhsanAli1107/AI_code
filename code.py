import streamlit as st
import requests

st.title("Chatbot Query Interface")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    if query:
        response = requests.post("http://localhost:8000/query", json={"query": query})
        if response.status_code == 200:
            st.write("Response:")
            st.write(response.json()["response"])
        else:
            st.write("Error:", response.status_code, response.text)
    else:
        st.write("Please enter a query.")
