# Import streamlit for creating a web-based UI for the chatbot
import streamlit as st

# Import functions to create a vector database and retrieve the Q&A chain
from helper import create_vector_db, get_qa_chain

# Set the title of the Streamlit app
st.title("Q and A Chatbot")

# Create a button labeled "Create Knowledgebase" to initialize the vector database
btn = st.button("Create Knowledgebase")

# If the button is clicked, call the create_vector_db function to generate the knowledge base
if btn:
    create_vector_db()

# Create a text input box for the user to enter a question
question = st.text_input("Question: ")

# If a question is entered, initialize the Q&A chain and get the response
if question:
    # Call get_qa_chain to retrieve the configured Q&A chain
    chain = get_qa_chain()

    # Pass the user's question to the chain and store the response
    response = chain(question)

    # Display the answer in the UI with a header and the result text
    st.header("Answer: ")
    st.write(response["result"])

