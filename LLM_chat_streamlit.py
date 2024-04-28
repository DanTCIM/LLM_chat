# Import
import streamlit as st
import os
import json
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts.prompt import PromptTemplate

# Setup page config and title
st.set_page_config(page_title="Prompt_Tester", page_icon="📖")
st.title("Test Prompts")
st.write("Try different prompts or make your own!")

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["ANTHROPIC_API_ID"] = st.secrets["ANTHROPIC_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# Define class
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


# Define variables for inputs
model_list = [
    "gpt-3.5-turbo",  # mid
    "gpt-4-turbo",  # large
    "claude-3-sonnet-20240229",  # mid
    "claude-3-opus-20240229",  # large
    "llama3-70b-8192",  # fast
]

model_dic = {
    "gpt-3.5-turbo": "OpenAI",
    "gpt-4-turbo": "OpenAI",
    "claude-3-opus-20240229": "Anthropic",
    "claude-3-sonnet-20240229": "Anthropic",
    "llama3-70b-8192": "Groq",
}


@st.cache_data
def load_prompts():
    with open("prompt.json") as f:
        prompts = json.load(f)
    # Extract the list of prompt names
    prompt_names = [p["prompt_name"] for p in prompts]

    # Create a dictionary to store the prompts
    prompts_dict = {p["prompt_name"]: p for p in prompts}

    return prompts_dict, prompt_names


prompts_dict, prompt_names = load_prompts()

with st.sidebar:
    with st.expander("⚙️ LLM setup", expanded=True):
        model_name = st.selectbox(
            "Select model",
            model_list,
        )

        prompt_name = st.selectbox(
            "Select system prompt",
            prompt_names,
        )
        selected_prompt = prompts_dict[prompt_name]
        st.write("➡️ " + selected_prompt["description"])

        user_system_message = st.text_area(
            label="System prompt",
            value=selected_prompt["prompt"],
            help="Feel free to update system prompt.",
        )

        user_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.25,
            help="Set to 0.0 for deterministic responses.",
        )

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
)

if model_dic[model_name] == "Anthropic":
    llm = ChatAnthropic(
        model=model_name,
        temperature=user_temperature,
        streaming=True,
    )
elif model_dic[model_name] == "Groq":
    llm = ChatGroq(
        model_name=model_name,
        temperature=user_temperature,
        streaming=True,
    )
else:
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=user_temperature,
        streaming=True,
    )

template = (
    user_system_message
    + """
The following is a conversation between a human and an AI.

Current conversation:
{history}
human: {input}
ai:"""
)
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

conversation_chain = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=memory,
)

# Initialize the chat history
if len(msgs.messages) == 0:
    msgs.add_ai_message("What is your question?")

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

# # Use a single block for handling different user prompts
# questions = [
#     "I have business concerns. Do you want to listen?",
#     "Can you analyze this policy proposal?",
#     "What digital marketing strategies should we employ for our campaign?",
#     "How can we optimize our election strategy?",
# ]

# # Generate buttons for each question
# for question in questions:
#     if st.sidebar.button(question):
#         with st.chat_message("user"):
#             st.write(question)
#         with st.chat_message("assistant"):

#             stream_handler = StreamHandler(st.empty())
#             response = conversation_chain.run(question, callbacks=[stream_handler])

with st.sidebar:

    def clear_chat_history():
        msgs.clear()
        msgs.add_ai_message("What is your question?")

    st.button(
        "Clear Chat",
        help="Clear chat history",
        on_click=clear_chat_history,
        use_container_width=True,
    )
