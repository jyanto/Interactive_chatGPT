# Import libraries
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

# Initialize session states
if "generated" not in st.session_state:
    st.session_state['generated'] = []
if "past" not in st.session_state:
    st.session_state['past'] = []
if "input" not in st.session_state:
    st.session_state['input'] = ""
if "stored_session" not in st.session_state:
    st.session_state['stored_session'] = []

def get_text():
    """
    Get the user input text
    """
    input_text = st.text_input("You: ", st.session_state['input'], key='input',
                               placeholder="Your AI assistant here! Ask me anything ...",
                               label_visibility='hidden')
    return input_text

def new_chat():
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state['past'][i])
        save.append("Bot:" + st.session_state['generated'][i])

    st.session_state['stored_session'].append(save)
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['input'] = ''
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()

st.title("Memory bot")

api = st.sidebar.text_input("API-Key", type='password')

llm_model = st.sidebar.selectbox(label='Model', options=['gpt-3.5-turbo', 'text-davinci-003'])

if api:
    llm = OpenAI(
        temperature=0.5,
        openai_api_key=api,
        model_name = llm_model
    )

    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=10)

    Conversation = ConversationChain(
        llm = llm,
        prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory = st.session_state.entity_memory
    )

else:
    st.error("No API found")

st.sidebar.button("New Chat", on_click=new_chat, type='primary')
user_input = get_text()

if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state['past'][i])
        st.success(st.session_state['generated'][i], icon='🤖')