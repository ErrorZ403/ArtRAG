import logging
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Message:
    content: str
    type: str
    metadata: Dict[str, Any] = None

class ChatMemoryManager:
    def __init__(self, max_context_len: int):
        self.logger = logging.getLogger("rag_chat.memory")
        self.logger.info(f"Initializing ChatMemoryManager with max_context_len: {max_context_len}")
        self.max_context_len = max_context_len
        self.msgs = StreamlitChatMessageHistory()
        self.memory = ConversationBufferMemory(
            chat_memory=self.msgs,
            return_messages=True,
            memory_key="chat_history",
            output_key="output",
            max_token_limit=self.max_context_len
        )
    
    def clear_memory(self):
        self.logger.info("Clearing chat memory")
        self.msgs.clear()
        self.msgs.add_ai_message("How can I help you?")
    
    def add_message(self, content: str, msg_type: str):
        self.logger.debug(f"Adding {msg_type} message: {content[:50]}...")
        if msg_type == "ai":
            self.msgs.add_ai_message(content)
        elif msg_type == "human":
            self.msgs.add_user_message(content)
    
    @property
    def messages(self):
        return self.msgs.messages 