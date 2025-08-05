import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os

st.set_page_config(page_title="DHT Aviation Assistant")
st.title("🛫 DHT Aviation Chatbot")
st.write("Trợ lý thông minh cho thủ tục hàng không tại Việt Nam.")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
persist_directory = "db"

if "qa" not in st.session_state:
    loader = DirectoryLoader("docs/", glob="**/*.docx", loader_cls=Docx2txtLoader)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    st.session_state.qa = qa

query = st.text_input("Bạn muốn hỏi gì? (VD: Visa cho phi hành đoàn, FBO, FAOC...)")

if query:
    with st.spinner("Đang xử lý..."):
        result = st.session_state.qa.run(query)
        st.write("### 💬 Trả lời:")
        st.write(result)