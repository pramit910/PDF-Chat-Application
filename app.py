"""Streamlit UI file for PDF Chatbot"""

import os
import sys
import pickle
from pathlib import PurePosixPath, PurePath
import logging
import uuid
from traceback import TracebackException, print_exception, print_exc, print_last
import logging
from logging import getLogger
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import StdOutCallbackHandler
from streamlit_extras.add_vertical_space import add_vertical_space


# initialize states
if 'pdf_text' not in st.session_state:
    st.session_state['pdf_text'] = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# set logging for PyPDF2
pdf_logger = logging.getLogger("PyPDF2")
pdf_logger.setLevel(logging.WARNING)
pdf_logger.setLevel(logging.ERROR)

st.set_page_config(page_title = "Home Page",
                   page_icon = ":notebook:",
                   layout = "centered",
                   initial_sidebar_state = "auto",
                   menu_items = None)

# sidebar contents for UI
with st.sidebar:
    st.title(":notebook: LLM Chat App")
    st.markdown("""
    ## About
    This is a PDF chat application.

    You can upload your files and all your questions about the PDF will be answered.
    """)
    add_vertical_space(5)
    st.write("Made by Pramit :balloon:")


@st.cache_resource
def initialize_llm(llm:str = "OpenAI"):
    """
    Initializes LLM, currently only OpenAI GPT 4o is used

    Args:
        llm: Name of the LLM service to be used. Ex: OpenAI, Gemini

    Returns:
        LLM which is ready for querying
    """
    if llm == "OpenAI":
        return ChatOpenAI(
            model='gpt-4o',
            temperature=0,
            verbose=True,
        )
    else:
        st.write("LLM Initialization failed. Please input correct name 'OpenAI' or use default Initialization!")
        return None

@st.cache_resource
def initialize_vector_store(embeddings, nlist:int, vector_store: str = "faiss"):
    try:
        if vector_store == 'faiss':
            # index creation for faiss
            d = len(embeddings.embed_query("hello world"))
            index = faiss.IndexFlatL2(d)
            # nlist = 5  # Num of voronoi cells
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)          

            # initialize vector store
            return index, FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
        else:
            st.write("Vector Store Initialization failed! Input valid parameters!")
    except Exception:
        st.error("Vector Store Initialization failed!", icon=":🚨:")
        print_exc()


def load_pdf(pdf_file):
    pass


def main():
    """
    Main applicationa logic
    """
    st.header("Chat with your PDFs :cloud:")

    add_vertical_space(3)

    # load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # file uploader
    pdf_file = st.file_uploader("Upload your PDF", type='pdf', accept_multiple_files=False) # single pdf file only
    if pdf_file is not None:
        pdf_reader = PdfReader(pdf_file)

        # extract text from the pdf
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # st.session_state.pdf_text = text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)
        # st.session_state.chunks = chunks

        # initialize everything again (the chunking and embedding process)
        embeddings = OpenAIEmbeddings() # initialize embedding model

        # initialize vector store
        index, vector_store = initialize_vector_store(embeddings, nlist=3)

        # generate uuids for knowledge base (indexing)
        uuids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        # train index for quantization if not trained already
        if not index.is_trained:
            text_embeddings = embeddings.embed_documents(chunks)
            index.train(np.array(text_embeddings))
        
        # construct knowledge base
        vector_store.add_texts(texts=chunks, ids=uuids)

        try:
            query = st.text_input("Ask any questions about your document:")
            if query:
                llm = initialize_llm()
                template = """
                You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Use three sentences maximum and keep the answer concise.
                Question: {question} 
                Context: {context} 
                Answer:
                """
                sys_prompt = ChatPromptTemplate.from_template(template)
                retriever = vector_store.as_retriever(search_kwargs = {'k':3})

                if llm is not None:
                    rag_chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | sys_prompt
                        | llm
                        | StrOutputParser()
                    )

                    # look at cost associated with each query -> callbacks
                    handler = StdOutCallbackHandler()
                    config = {
                        'callbacks' : [handler]
                    }

                    # generate step
                    answer = rag_chain.invoke(query, config=config)
                    st.markdown(answer) # single answer for now

        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = TracebackException(exc_type, exc_value, exc_tb)
            print(''.join(tb.format_exception_only()))
            

if __name__ == "__main__":
    main()
