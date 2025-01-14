import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory

import os

os.environ['OPENAI_API_KEY'] = 'dummy_api_key'

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


st.set_page_config(page_title="Story GPT", page_icon="📖")
st.title("Welcome! Story GPT")

with st.sidebar:
    st.page_link(page="https://github.com/kjy7097/gpt_fullstack_assignment6.git",label="Github Repo.")
    api_key = st.text_input("Enter OpenAI API Key....")
    if api_key:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
        )

if api_key:
    st.markdown(
        """
                Please upload a story book. 

                Ask me any question about the book."""
    )
    os.environ['OPENAI_API_KEY'] = api_key
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )
else:
    st.markdown(
        """
                Please enter API key first.
        """
    )



if not "history" in st.session_state:
    st.session_state["history"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def save_message(message, role):
    st.session_state["history"].append(
        {
            "message": message,
            "role": role,
        }
    )


def print_history():
    for history_data in st.session_state["history"]:
        send_message(history_data["message"], history_data["role"], False)


@st.cache_data(show_spinner="Embedding file....")
def embed_file(file):
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f".cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cache_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_data()
def get_memory(file):
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
    )
    return memory


if not "memory" in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
    )
else:
    memory = st.session_state["memory"]


def map_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def load_memory(_):
    return memory.load_memory_variables({})["history"]


def invoke_chain(message, input_chain):
    result = input_chain.invoke(message)
    st.session_state["memory"].save_context(
        {"input": message}, {"output": result.content}
    )
    return result.content


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up,
            --------
            {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

if api_key:
    if file:
        retriever = embed_file(file)
        send_message("I am ready to answer!", "ai", save=False)
        chain = (
            {
                "context": retriever | RunnableLambda(map_docs),
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough.assign(history=load_memory)
            | prompt
            | llm
        )
        message = st.chat_input("Ask anything about the book...")
        if message:
            if len(st.session_state["history"]) > 0:
                print_history()
            send_message(message, "Human")
            with st.chat_message("ai"):
                invoke_chain(message, chain)
    else:
        st.session_state["history"] = []
