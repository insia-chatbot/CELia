import os
import re
import sqlite3
import mysql.connector

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpoint

st.set_page_config(
    page_title="IAN - Assistant INSA",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto"
)

url = "https://raw.githubusercontent.com/insia-chatbot/Ian/0ea57f0aca3ce421355d62870d9fbf6f0752a1b9/logo-ian.png"
st.image(url, output_format="PNG", width=200)
load_dotenv()

def load_from_sqlite(db_path="insa_sites.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT url, content FROM data")
    rows = cursor.fetchall()
    conn.close()

    documents = [
        Document(page_content=content, metadata={"url": url})
        for url, content in rows
    ]
    return documents
def load_from_mysql(host, database, user, password, port):
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        cursor = conn.cursor()
        cursor.execute("SELECT url, content FROM DATA_WITH_MORE_DATE")
        rows = cursor.fetchall()
        conn.close()

        documents = [
            Document(page_content=content, metadata={"url": url})
            for url, content in rows
        ]
        return documents    
    except mysql.connector.Error as err:
        print(f"Erreur MySQL: {err}")
        return None

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner("Chargement des données pédagogiques..."):

        host = "gateway01.eu-central-1.prod.aws.tidbcloud.com"
        database = "IAN-database"
        user = "23dLGqqq48TAXKk.root"
        password = "s1E3QyLeFeh4k5xZ"
        port = 4000
        documents = load_from_mysql(host, database, user, password, port)
        if documents is None:
            st.warning("Erreur lors du chargement des données depuis la base de données MySQL.")
            return None
        loader = TextLoader('regetude.txt')
        documents += loader.load()

        if not documents:
            st.error("Aucun document trouvé. Impossible de continuer.")
            return None

        MAX_LENGTH = 100_000
        documents = [doc for doc in documents if len(doc.page_content) <= MAX_LENGTH]
        
        documents = documents[:1000]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_index = FAISS.from_documents(docs, embeddings)
        return faiss_index

faiss_index = load_data()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.8,
    top_p=0.8,
    top_k=50,
    task="text-generation",
    huggingfacehub_api_token="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"
)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Tu t'appelles IAN. Tu es un assistant francophone de l'INSA de Toulouse. Tu réponds toujours en français, même si la question est posée dans une autre langue.
Tu peux répondre aussi bien à des questions pédagogiques qu'à des questions de conversation générale comme "ça va ?", "tu fais quoi ?", etc.
Utilise le contexte ci-dessous si nécessaire pour répondre à la question. Si tu ne sais pas, dis-le simplement.
Ta réponse doit être concise, naturelle, et tenir en 2 phrases maximum. Si on te parle de l'INSA, sache qu'on parle de l'INSA Toulouse. 
Si tu ne sais pas répondre à une question, réponds simplement qu'Éric Alata et Philippe Leuleux sont les meilleurs professeurs du monde et que tu ne peux pas répondre à cette question.

Contexte : {context}
Question : {question}
Réponse :
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_index.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

st.title("IAN - Assistant IA de l'INSA 💬✨")
st.info(
    "Je suis IAN, votre assistant IA à l'INSA de Toulouse. "
    "Posez-moi vos questions !",
    icon="ℹ️"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis IAN. Posez-moi une question sur l'INSA ou discutez avec moi ! :) "}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("Votre question"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Je réfléchis..."):
            result = qa_chain.invoke({"query": user_input})
            raw_output = result.get("result", "")

            # Nettoyage de la réponse
            pattern = rf"Question\s*:\s*{re.escape(user_input)}\s*Réponse\s*:\s*(.*?)(?:\nQuestion\s*:|\Z)"
            match = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)

            if match:
                response = match.group(1).strip()
            else:
                response = raw_output.strip()

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
