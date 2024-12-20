import logging
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from typing import List
from config.config import FaissDBConfig

class DocumentRetriever:
    def __init__(self, faiss_config: FaissDBConfig):
        self.logger = logging.getLogger("rag_chat.retriever")
        self.logger.info("Initializing DocumentRetriever")
        self.embeddings = OpenAIEmbeddings(openai_api_key=faiss_config.openai_key)
        self.vectorstore = None
        self.prompt = ChatPromptTemplate.from_template("""
            Use the following pieces of context to answer the question at the end.
            If the context doesn't provide enough information, just say that you don't know, don't try to make up an answer.
            Pay attention to the context of the question rather than just looking for similar keywords in the corpus.
            Keep the answer as concise as possible.
            Always say "thanks for asking!" at the end of the answer.
            {context}
            Question: {question}
            Helpful Answer:
        """)

    def ingest_documents(self, documents: List[str]):
        self.logger.info(f"Ingesting {len(documents)} documents")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        self.logger.info("Documents successfully ingested into vector store")

    def get_relevant_documents(self, query: str, k: int = 4):
        if not self.vectorstore:
            self.logger.warning("Vector store is not initialized")
            return []
        self.logger.debug(f"Retrieving {k} relevant documents for query: {query[:50]}...")
        return self.vectorstore.similarity_search(query, k=k)

    def get_prompt_messages(self, question: str):
        self.logger.debug(f"Generating prompt messages for question: {question[:50]}...")
        relevant_docs = self.get_relevant_documents(question)
        return self.prompt.format_messages(context=relevant_docs, question=question) 