import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from streamlit_agent.memory_storage import ChatMemoryManager
from streamlit_agent.retriever import DocumentRetriever
from config.config import load_config, AiChatModel
from config.logging_config import setup_logging
from openai import AzureOpenAI

st.set_page_config(page_title="RAG-powered Chat", page_icon="🤖")

class ChatApplication:
    def __init__(self):
        # st.set_page_config(page_title="RAG-powered Chat", page_icon="🤖")
        self.logger = setup_logging()
        self.logger.info("Initializing Chat Application")
        self.config = load_config()
        self.memory_manager = ChatMemoryManager(
            max_context_len=self.config.chat_model.chatbot.max_context_len
        )
        self.setup_sidebar()

        self.retriever = self.initialize_retriever()
        self.llm = self.initialize_llm()

    def setup_sidebar(self):
        st.sidebar.title("Configuration")
        self.openai_api_key = self.config.faiss_config.openai_key
        if st.sidebar.button("Reset chat history"):
            self.logger.info("Clearing chat history")
            self.memory_manager.clear_memory()
            st.session_state.steps = {}

    @st.cache_resource
    def initialize_retriever(_self):
        return DocumentRetriever(_self.config.faiss_config)

    @st.cache_resource
    def initialize_llm(_self):
        return AzureOpenAI(
            api_version="2024-02-15-preview",
            azure_endpoint="https://utbd.openai.azure.com",
            api_key=_self.openai_api_key,
        )

    def display_chat_history(self):
        avatars = {"human": "user", "ai": "assistant"}
        for msg in self.memory_manager.messages:
            with st.chat_message(avatars[msg.type]):
                st.write(msg.content)

    def run(self):
        st.title("🤖 RAG-powered Chat Assistant")

        self.display_chat_history()

        if prompt := st.chat_input(placeholder="Ask me anything!"):
            self.logger.info(f"Received user prompt: {prompt[:50]}...")
            st.chat_message("user").write(prompt)
            self.memory_manager.add_message(prompt, "human")

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

                try:
                    messages = self.retriever.get_prompt_messages(prompt)
                    response = self.llm.chat.completions.create(
                        model="gpt4_small", messages=messages, n=1, temperature=0.4, max_tokens=300
                    )
                    response = response.choices[0].message.content
                    st.write(response)
                    self.memory_manager.add_message(response, "ai")
                    self.logger.info("Successfully generated and displayed response")
                except Exception as e:
                    self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
                    st.error("An error occurred while generating the response.")


if __name__ == "__main__":
    app = ChatApplication()
    app.run()