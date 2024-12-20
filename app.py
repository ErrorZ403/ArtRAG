import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler
from streamlit_agent.memory_storage import ChatMemoryManager
from streamlit_agent.retriever import DocumentRetriever
from config.config import load_config, AiChatModel
from config.logging_config import setup_logging

class ChatApplication:
    def __init__(self):
        st.set_page_config(page_title="RAG-powered Chat", page_icon="🤖")
        self.logger = setup_logging()
        self.logger.info("Initializing Chat Application")
        self.config = load_config()
        self.setup_sidebar()
        self.memory_manager = ChatMemoryManager(
            max_context_len=self.config.chat_model.chatbot.max_context_len
        )
        self.retriever = None
        self.llm = None

    def setup_sidebar(self):
        st.sidebar.title("Configuration")
        self.openai_api_key = self.config.faiss_config.openai_key
        if st.sidebar.button("Reset chat history"):
            self.logger.info("Clearing chat history")
            self.memory_manager.clear_memory()
            st.session_state.steps = {}

    def initialize_components(self):
        if not self.openai_api_key:
            self.logger.error("OpenAI API key not configured")
            st.info("Please check your OpenAI API key configuration.")
            return False

        if not self.llm:
            self.logger.info("Initializing LLM and Document Retriever")
            model_params = self.config.chat_model.get_genai_chat_params()
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                streaming=True,
                **model_params
            )
            self.retriever = DocumentRetriever(self.config.faiss_config)
        return True

    def display_chat_history(self):
        avatars = {"human": "user", "ai": "assistant"}
        for msg in self.memory_manager.messages:
            with st.chat_message(avatars[msg.type]):
                st.write(msg.content)

    def run(self):
        st.title("🤖 RAG-powered Chat Assistant")
        
        if not self.initialize_components():
            return

        self.display_chat_history()

        if prompt := st.chat_input(placeholder="Ask me anything!"):
            self.logger.info(f"Received user prompt: {prompt[:50]}...")
            st.chat_message("user").write(prompt)
            self.memory_manager.add_message(prompt, "human")

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                
                try:
                    messages = self.retriever.get_prompt_messages(prompt)
                    response = self.llm.predict(messages, callbacks=[st_cb])
                    
                    st.write(response)
                    self.memory_manager.add_message(response, "ai")
                    self.logger.info("Successfully generated and displayed response")
                except Exception as e:
                    self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
                    st.error("An error occurred while generating the response.")

if __name__ == "__main__":
    app = ChatApplication()
    app.run() 