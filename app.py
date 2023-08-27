import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'


custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    print(embeddings)

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


def clean_up():
    # Delete all files in the data folder
    data_folder = "data"
    if os.path.exists(data_folder):
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            os.remove(file_path)

    # Delete the vectorstore folder if it exists
    # if os.path.exists(DB_FAISS_PATH):
    #     shutil.rmtree(DB_FAISS_PATH)


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(
                                               search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm


def get_conversation_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


def handle_user_input(user_question):
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        qa_chain = get_conversation_chain()
        st.session_state.conversation = qa_chain

    response = st.session_state.conversation(
        {'question': user_question, 'query': user_question})

    if response and 'result' in response:
        with st.chat_message("assistant"):
            st.write(response['result'])


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat-Bot",
                       page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    clean_up()

    st.header("Chat-Bot :books:")
    user_question = st.chat_input("Ask a question about your document")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                if pdf_docs:
                    for pdf_file in pdf_docs:
                        pdf_file_name = pdf_file.name
                        pdf_file_path = os.path.join("data", pdf_file_name)
                        with open(pdf_file_path, "wb") as f:
                            f.write(pdf_file.read())
                    create_vector_db()


if __name__ == '__main__':
    main()
