import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## function to extract all the texts from all the pdfs and store in a variable, kindof make all the pdfs texts as a form a bigger prompt input

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs: 
        pdf_reader = PdfReader(pdf)
        ## pdf_reader contains all text in the form of pages
        for page in pdf_reader.pages:
            text+=page.extract_text()
    
    return text 
    ## returns a large string 

## function to take the text from get_pdf_text and convert it into chunks of x characters/tokens

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks 
    ## here chunks is an array/list of 10000 characters/tokens


## Converting the chunks in vectors which are not human readable

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    
    ## these vector_store can be stored in a db or locally also
    ## storingthem locally in a folder called "faiss_index" or "<your_choice"
    vector_store.save_local("faiss_index")


## defining the prompt template with input variables

def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, is the answer is not in the provided context just say, "answer is not available in the context/pdfs", don't provide any wrong answers \n \n
    Context: \n {context}? \n
    Question: \n {question} \n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


## function to take userinput and return the response

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        },
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


## streamlit frontend

def main():
    st.set_page_config("Chat w multiple PDFs")
    st.header("Chat with multiple PDFs using Gemini")

    user_question = st.text_input("Ask a Question related to uploaded pdfs")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDFs and click on Submit & Process button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()