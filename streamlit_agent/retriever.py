import logging
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from typing import List
from config.config import FaissDBConfig


class DocumentRetriever:
    def __init__(self, faiss_config: FaissDBConfig):
        self.logger = logging.getLogger("rag_chat.retriever")
        self.logger.info("Initializing DocumentRetriever")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"batch_size": 1},
        )
        self.vectorstore = FAISS.load_local(
            "data/e5_wikiart_small",
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def ingest_documents(self, documents: List[str]):
        self.logger.info(f"Ingesting {len(documents)} documents")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        self.logger.info("Documents successfully ingested into vector store")

    def get_relevant_documents(self, query: str, k: int = 5):
        if not self.vectorstore:
            self.logger.warning("Vector store is not initialized")
            return []
        self.logger.debug(f"Retrieving {k} relevant documents for query: {query[:50]}...")
        return self.vectorstore.similarity_search(query, k=k)

    def get_prompt_messages(self, question: str):
        self.logger.debug(f"Generating prompt messages for question: {question[:50]}...")
        relevant_docs = self.get_relevant_documents(question)

        messages = [
            {
                "role": "user",
                "content": f"""Use the following pieces of context to answer the question at the end.
                If the context doesn't provide enough information, you can generate your own answer based on your knowledge, but don't say the answer isn't in context in such a case. If you are not sure of the correctness of your answer, say that you don't know the answer to the question..
                Pay attention to the context of the question rather than just looking for similar keywords in the corpus.
                Keep the answer as concise as possible.
                Always say "thanks for asking!" at the end of the answer.
                {relevant_docs}
                Question: {question}
                Helpful Answer:""",
            }
        ]

        return messages
