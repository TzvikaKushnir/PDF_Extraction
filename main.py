# third party imports
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai.embeddings import OpenAIEmbeddings

# python internal imports
import os
import shutil

# local imports
from config import mll_config as config
from responser import Responser

load_dotenv()
text_splitter = config.SPLITTER


def extract_pdf_into_documents(pdf_folder: str) -> list[Document]:
    loader = DirectoryLoader(pdf_folder, glob='*.pdf', show_progress=True, use_multithreading=True)
    documents = loader.load()

    return documents


def document_analyze(document: Document) -> list:
    # Step2 - Spliting the document into chunks
    chaunked_document = text_splitter.split_documents([document])
    qna_bot = Responser(config.CHROMA_PATH, chaunked_document, OpenAIEmbeddings())
    # Step3 - Embeddings the chaunkd document AND Creating the VectorDB
    qna_bot.save_to_chroma()
    # Step4 - 
    res = qna_bot.get_answers(config.QUERIES)

    return res


if __name__ == "__main__":
    # Step1 - Loading the PDF files as a documents class
    documents_list = extract_pdf_into_documents(config.INPUT_PDFS)

    # Clear out the database first
    if os.path.exists(config.CHROMA_PATH):
        shutil.rmtree(config.CHROMA_PATH)

    results_list = []
    for docu in documents_list:
        result = document_analyze(docu)
        results_list.append(result)
