from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader


def extract_text_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_pdf(file_path):
    text = extract_text_pdf(file_path)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks

def chunk_txt(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    return chunks

def embed_doc(file_path, db_dir):
    if file_path[-3:] == "pdf":
        chunks = chunk_pdf(file_path)
    else:
        chunks = chunk_txt(file_path)
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=db_dir)
    return db

def create_template(conversation_history):
    template = """You are a member of a book club engaging in question-ansering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer consice.
    After answering, ask a follow-up question to that dives deeper into the context 
    if appropriate.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return template, prompt, llm
