import streamlit as st
from pathlib import Path
from langchain.llms.openai import OpenAI
from langchain.agents import create_sql_agent, ConversationalChatAgent, AgentExecutor
from langchain.sql_database import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
import sqlite3

st.set_page_config(page_title="LangChain: Combined RAG and Web Search", page_icon="")
st.title("LangChain: Combined RAG and Web Search")

LOCALDB = "USE_LOCALDB"

@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    if db_uri == LOCALDB:
        db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    else:
        return SQLDatabase.from_uri(db_uri)

db_uri = st.sidebar.text_input("Database URI", value=LOCALDB)
db = configure_db(db_uri)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.error("Please enter your OpenAI API key.")
    st.stop()

llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, streaming=True)
search_tool = DuckDuckGoSearchRun()

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
callback_handler = StreamlitCallbackHandler()

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools() + [search_tool]

agent = ConversationalChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    memory=memory,
    callbacks=[callback_handler],
)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I assist you today?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for i, msg in enumerate(msgs.messages):
    st.chat_message(avatars[msg.type]).write(msg.content)

if prompt := st.chat_input():
    msgs.add_user_message(prompt)
    with st.chat_message("assistant"):
        response = agent_executor.run(prompt, callbacks=[callback_handler])
        st.write(response)
        msgs.add_ai_message(response)
