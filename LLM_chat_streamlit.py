# Import
import streamlit as st
from openai import OpenAI
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Setup title
st.title("Management & Political Consulting")
st.write("Structured, creative problem-solving with user collaboration")

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Define variables for inputs
default_system_message = "You are a helpful assistant."
model_name = "gpt-4-turbo-preview"  ##gpt-3.5-turbo

user_system_message = st.sidebar.text_area(
    label="System Instruction",
    value=default_system_message,
    help="Enter your system instructions here.",
)
user_temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.0,
    step=0.25,
    help="Set to 0.0 for deterministic responses.",
)


# Define a function to get completion based on user input
## UNUSED
def get_completion(user_input, system_message=user_system_message):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": user_input},
        ],
        temperature=user_temperature,
    )
    return completion.choices[0].message.content


# Define functions to get completion based on user input and history of chat
def transform_messages(messages):
    transformed_list = []
    for message in messages:
        if message["role"] == "user":
            transformed_list.append(HumanMessage(content=message["content"]))
        elif message["role"] == "ai":
            transformed_list.append(AIMessage(content=message["content"]))
    return transformed_list


def get_completion_history(
    user_input,
    session_state=[],
    system_message=user_system_message,
):

    chat = ChatOpenAI(model=model_name, temperature=user_temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | chat
    message_history = []
    message_history = transform_messages(session_state)
    message_history.append(HumanMessage(content=user_input))

    message_dict = {
        "messages": message_history,
    }

    ai_message = chain.invoke(message_dict)
    return ai_message.content
    # yield chain.stream(message_dict)


# Initialize chat history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "ai", "content": "What is your question?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User-provided prompt
if user_prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt, "type": "text"}
    )
    with st.chat_message("user"):
        st.write(user_prompt)

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
        user_prompt = question
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt, "type": "text"}
        )
        with st.chat_message("user"):
            st.write(user_prompt)

with st.sidebar:

    def clear_chat_history():
        st.session_state.messages = [
            {
                "role": "ai",
                "content": "What is your question?",
            }
        ]

    st.button(
        "Clear Chat",
        help="Clear chat history",
        on_click=clear_chat_history,
        use_container_width=True,
    )

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "ai":
    with st.chat_message("ai"):
        with st.spinner("GPT-4 running..."):
            response_generator = get_completion_history(
                user_prompt,
                session_state=st.session_state.messages,
                system_message=user_system_message,
            )

            st.write(response_generator)

    message = {"role": "ai", "content": response_generator}
    st.session_state.messages.append(message)
