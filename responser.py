# third party imports
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# python internal imports
from dataclasses import dataclass, field
from typing import TypeVar

# special types for this class
Embeddings = TypeVar('Embeddings')


@dataclass
class Responser:
    chroma_path         : str
    chaunked_documents  : list[Document]
    embedding_model     : Embeddings = OpenAIEmbeddings() # to do diffult or not??
    vec_db              : Chroma = field(init=False, repr=False, hash=False)

    def save_to_chroma(self) -> None:

        # Create a new VectorDB from the givan document
        self.vec_db = Chroma.from_documents(
            self.chaunked_documents,
            embedding=self.embedding_model,
            persist_directory=self.chroma_path
        )
        self.vec_db.persist()

    def get_answers(self, queries: list[str]) -> list:
        qna_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo-0125'),
            retriever=self.vec_db.as_retriever(search_kwargs={'k':6}),
            return_source_documents=True,
            verbose=False
        )

        chat_history = []
        for query in queries:
            result = qna_chain({'question': query, 'chat_history': chat_history})
            chat_history.append((query, result['answer']))

        self.vec_db.delete_collection()

        return chat_history