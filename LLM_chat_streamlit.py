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

# Define variables for inputs
model_list = [
    # "gpt-3.5-turbo",  # mid
    "claude-3-5-sonnet-20240620",  # large
    "gpt-4o",  # large
    "llama3-70b-8192",  # fast
]

model_dic = {
    "claude-3-5-sonnet-20240620": "Anthropic",
    "gpt-4o": "OpenAI",
    "llama3-70b-8192": "Groq",
}


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


@st.cache_data
def load_prompts():
    with open("prompt.json") as f:
        prompts = json.load(f)
    prompt_names = [p["prompt_name"] for p in prompts]
    prompts_dict = {p["prompt_name"]: p for p in prompts}
    return prompts_dict, prompt_names


def setup_model_param(prompts_dict, prompt_names):
    with st.sidebar:
        with st.expander("**⚙️ LLM setup**", expanded=True):
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
                height=300,
            )

            user_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.25,
                help="Set to 0.0 for deterministic responses.",
            )
    return model_name, user_system_message, user_temperature


def setup_llm(model_name, user_temperature):
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
    return llm


def setup_conversation_chain(user_system_message, llm, memory):
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

    return conversation_chain


def display_chat_history(msgs):
    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)


def handle_user_query(conversation_chain):
    if user_query := st.chat_input(placeholder="What is your question?"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):

            stream_handler = StreamHandler(st.empty())
            response = conversation_chain.run(user_query, callbacks=[stream_handler])


def clear_chat_history(msgs):
    msgs.clear()
    msgs.add_ai_message("What is your question?")


def clear_chat_button(msgs):
    with st.sidebar:
        st.button(
            "Clear Chat",
            help="Clear chat history",
            on_click=lambda: clear_chat_history(msgs),
            use_container_width=True,
        )


def sidebar_faq():
    with st.sidebar:
        with st.expander("**FAQ**", expanded=True):
            st.write(
                "**Llama 3:** Meta's open-source flagship model. Llama3 is deployed by Groq (https://groq.com/), showcasing its impressive speed."
            )
            st.write("**GPT-4o:** OpenAI's flagship model.")
            st.write(
                "**Claude-3.5-Sonnet:** Anthropic's flagship model with a 200K input window size."
            )
            st.write(
                "**System prompts:** The examples are from Anthropic's prompt library. Visit https://docs.anthropic.com/claude/prompt-library for more examples."
            )


def main():
    st.set_page_config(page_title="LLM Playground", page_icon="📖")
    st.title("Large Language Model and Prompt Playground")
    st.write("Try different LLMs and prompts sourced from Anthropic's library!")

    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_ID"] = st.secrets["ANTHROPIC_API_KEY"]
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    prompts_dict, prompt_names = load_prompts()
    model_name, user_system_message, user_temperature = setup_model_param(
        prompts_dict, prompt_names
    )

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs,
        return_messages=True,
    )

    llm = setup_llm(model_name, user_temperature)
    conversation_chain = setup_conversation_chain(user_system_message, llm, memory)

    # Initialize the chat history
    if len(msgs.messages) == 0:
        msgs.add_ai_message("What is your question?")

    display_chat_history(msgs)
    handle_user_query(conversation_chain)

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

    clear_chat_button(msgs)
    sidebar_faq()


if __name__ == "__main__":
    main()
