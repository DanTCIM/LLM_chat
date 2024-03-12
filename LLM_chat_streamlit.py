# Import
import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler

# Setup title
st.title("Management & Political Consulting")
st.write("Structured, creative problem-solving with user collaboration")

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Define variables for inputs
default_system_message = "You are a helpful assistant."
model_list = ["gpt-4-turbo-preview", "gpt-3.5-turbo"]

with st.sidebar:
    with st.expander("⚙️ LLM setup"):
        model_name = st.selectbox(
            "Select model",
            model_list,
            help="GPT3.5 is way faster than GPT-4",
        )
        user_system_message = st.text_area(
            label="System Instruction",
            value=default_system_message,
            help="Enter your system instructions here.",
        )
        user_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.25,
            help="Set to 0.0 for deterministic responses.",
        )


# Set up conversation chain
class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
)

llm = ChatOpenAI(
    model_name=model_name,
    temperature=user_temperature,
    streaming=True,
)
conversation_chain = ConversationChain(llm=llm, verbose=True, memory=memory)


# Initialize the chat history
def clear_chat_history():
    msgs.clear()
    msgs.add_ai_message("What is your question?")


if len(msgs.messages) == 0:
    clear_chat_history()

# Show the chat history
avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# User asks a question
if user_query := st.chat_input(placeholder="What is your question?"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = conversation_chain.run(user_query, callbacks=[stream_handler])

# Use a single block for handling different user prompts
questions = [
    "I have business concerns. Do you want to listen?",
    "Can you analyze this policy proposal?",
    "What digital marketing strategies should we employ for our campaign?",
    "How can we optimize our election strategy?",
]

# Generate buttons for each question
for question in questions:
    if st.sidebar.button(question):
        user_query = question
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = conversation_chain.run(question, callbacks=[stream_handler])


with st.sidebar:
    st.button(
        "Clear Chat",
        help="Clear chat history",
        on_click=clear_chat_history(),
        use_container_width=True,
    )
