import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from langchain.llms import CTransformers


import os
import pickle

# sidebar contents
with st.sidebar:
    st.title("LLM CHATBOT")
    st.markdown(
        '''
        ## About
        This app is an LLM based chatbot using:
        - streamlit
        - LangChain
        - HuggingFace
        '''
    )
    add_vertical_space(5)
    st.write('Made by Shrijana')

def main():
    st.header("CHAT with PDF")

    load_dotenv()

    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = "normal"

    if "user_info" not in st.session_state:
        st.session_state.user_info = {"name": "", "phone": "", "email": ""}

    def collect_user_info():
        st.session_state.user_info["name"] = st.text_input("Please enter your name:", key="name")
        st.session_state.user_info["phone"] = st.text_input("Please enter your phone number:", key="phone")
        st.session_state.user_info["email"] = st.text_input("Please enter your email:", key="email")
        if st.session_state.user_info["name"] and st.session_state.user_info["phone"] and st.session_state.user_info["email"]:
            st.session_state.conversation_state = "normal"
            st.write(f"Thank you, {st.session_state.user_info['name']}! We will contact you at {st.session_state.user_info['phone']} or {st.session_state.user_info['email']}.")

    # to upload a pdf
    pdf = st.file_uploader("Upload your pdf", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        #st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        #st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 300,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)
        #st.write(chunks)

        #embeddings

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            #st.write('Embeddings Loaded from the disk')
        else: 
            api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
           
            embeddings = HuggingFaceEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
            
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            #st.write('Embeddings Completed')

        # To accept user queries
        query = st.text_input("Ask questions about the PDF file:")
        #st.write(query)

        if query:
            if "call me" in query.lower():
                st.session_state.conversation_state = "collect_info"

            if st.session_state.conversation_state == "collect_info":
                collect_user_info()
            else:
                # Using Hugging Face pipeline for question answering
                qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2', tokenizer='deepset/roberta-base-squad2')
                result = qa_pipeline(question=query, context=text)
                st.write(result['answer'])
                #docs = VectorStore.similarity_search(query=query, k=3)
                #llm = HuggingFaceHub(repo_id = "deepset/roberta-base-squad2", model_kwargs= {"temperature":0, "max_length":512})
                #chain = load_qa_chain(llm, chain_type="stuff")

                #chain.run(input_documents = docs, question = query )
            

if __name__ == '__main__':
    main()