# Import getpass to securely prompt for sensitive input, like API keys
from getpass import getpass
from dotenv import load_dotenv

# Import RetrievalQA to create a question-answering chain for information retrieval
from langchain.chains.retrieval_qa.base import RetrievalQA

# Import PromptTemplate to define custom prompts for the language model
from langchain_core.prompts import PromptTemplate

# Import GoogleGenerativeAI and GoogleGenerativeAIEmbeddings to use Google’s AI models
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import FAISS, a vector store for efficient similarity search and clustering of dense embeddings
from langchain_community.vectorstores import FAISS

# Import CSVLoader to load documents from CSV files for processing
from langchain_community.document_loaders.csv_loader import CSVLoader

import os
load_dotenv()


def get_api_key():
    # Retrieve the API key from environment variables or prompt the user if it's not found
    try:
        return os.getenv("GOOGLE_API_KEY")
    except Exception:
        return getpass("Please enter your Google API key:")


api_key = get_api_key()

llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, max_tokens=500)

instructor_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)

# path to save or load the FAISS vector database
vectordb_file_path = "faiss_index"


def create_vector_db():
    loader = CSVLoader(file_path='faqs.csv', source_column='prompt')

    docs = loader.load()

    # Create a FAISS vector store using the loaded documents and embeddings
    vectordb = FAISS.from_documents(documents=docs, embedding=instructor_embeddings)

    # Save the vector database locally to the specified path
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from local storage, allowing safe deserialization
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Set up a retriever for querying the vector database with a defined score threshold
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Define a prompt template to guide the language model's response to queries
    prompt_template = """
    Given the following context and a question, generate an answer based on this context only.
    In the answer, try to provide as much text as possible from the "response" section in the source document context without making major changes.
    If the answer is not found in the context, just state "I don't know." Don't attempt to generate an answer.

    CONTEXT: {context}
    QUESTION: {question}
    """

    # Initialize the PromptTemplate with the custom prompt and required variables
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create a RetrievalQA chain to use the language model with the defined retriever and prompt
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain


