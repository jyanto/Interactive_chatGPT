# Import libraries
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor
from langchain.agents import Tool
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
from langchain.agents import initialize_agent
import openai

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

# @st.cache_data
# def create_indexing(data):
#     documents = SimpleDirectoryReader("./").load_data()
#     index = GPTSimpleVectorIndex(documents)
#     return index

def load_indexing(data):
    data = data.read()
    index = GPTSimpleVectorIndex.load_from_string(data)
    return index

## Extract the intent entity
# def extract_intent(data):
#     prompt = f"Intent entity dari {data}"
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo", 
#         messages = [{"role": "system", "content" : "Kamu adalah ChatGPT"},
#         {"role": "user", "content" : prompt}],
#         temperature=0.5,
#         max_tokens=256,
#     )
#     description = "Berguna untuk menjawab pertanyaan tentang " + completion.choices[0].message.content
#     return description

def new_chat():
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state['past'][i])
        save.append("Bot:" + st.session_state['generated'][i])

    st.session_state['stored_session'].append(save)
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['input'] = ''
    # st.session_state.conv_memory.store = {}
    st.session_state.conv_memory.buffer.clear()
    st.session_state.conv_memory = []

# Set the title
st.title("chatGPT + Langchain + LlamaIndex")

# Set the box to input api-key
api = st.sidebar.text_input("API-Key", type='password')

# Set the selectbox for user to select the llm model version
llm_model = st.sidebar.selectbox(label='Model', options=['gpt-3.5-turbo', 'text-davinci-003'])

# Set the prompt memory for conversational
prompt_memory = st.sidebar.number_input(label='Prompt memory', min_value=3, max_value=15)

# Set the data uploader for external database
uploaded_data = st.sidebar.file_uploader(
    "Upload indexing file (data in .json format)", accept_multiple_files=False
)
uploaded = False # Set default to False (normal chatGPT without querying external data)

if uploaded_data:
    with st.spinner("Processing data ..."):
        index = load_indexing(uploaded_data)
    # Let the user to input the intent entity manually for now
    intent_entity = st.sidebar.text_input("Intent entity of uploaded data")
    description = f"Berguna untuk menjawab pertanyaan tentang {intent_entity}"
    llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, n=1))
    tools = [Tool(
                name = "GPT Index",
                func=lambda q: str(index.query(q, optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.5), llm_predictor=llm_predictor)),
                description=description,
                return_direct=True
                    )
            ]
    uploaded = True

if api:
    #Setup the llm
    llm = OpenAI(
        temperature=0.5,
        openai_api_key=api,
        model_name = llm_model
    )
    if uploaded:
        #Setup the conversational memory
        # if 'conv_memory' not in st.session_state:
        st.session_state.conv_memory = ConversationBufferWindowMemory(k=prompt_memory, memory_key="chat_history")
        # Set the agent tools if the external database is uploaded
        agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", memory=st.session_state.conv_memory)
    else:
        #Setup the conversational memory
        # if 'conv_memory' not in st.session_state:
        st.session_state.conv_memory = ConversationEntityMemory(llm=llm, k=prompt_memory)
        # Set the conversation agent tools if external database is not uploaded
        agent_chain = ConversationChain(
            llm = llm,
            prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory = st.session_state.conv_memory
        )

else:
    st.error("No API found")

st.sidebar.button("New Chat", on_click=new_chat, type='primary')
user_input = get_text()

if user_input:
    output = agent_chain.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state['past'][i])
        st.success(st.session_state['generated'][i], icon='🤖')