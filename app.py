import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def getPdfText(pdf_docs):
   text = ""
   for pdf in pdf_docs:
      pdfReader = PdfReader(pdf)
      for page in pdfReader.pages:
         text += page.extract_text()

   return text

def getTextChunks(raw_text, chunk_size=1000, overlap=200):
   textSplitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=chunk_size,
      chunk_overlap=overlap,
      length_function=len
   )
   textChunks = textSplitter.split_text(raw_text)
   return textChunks

def getVectorStore(textChunks):
    print("Function getVectorStore")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=gemini_api_key)
   #  embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_texts(texts=textChunks,embedding= embeddings)
    print("Stored in the vector store")
    return vectorStore

   #  embeddings= HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

def getConversationChain(vectorStore):
   memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
   # We need the memory.
   #We need the language model.
   conversation_chain=ConversationalRetrievalChain.from_llm()
   
   
def main():
   load_dotenv()  # Load environment variables from .env file
   st.set_page_config(page_title="Chat with PDFs",page_icon="books")
   st.header("Chat with multiple PDFs: books,articles")
   st.text_input("Ask a question about your docs:")

   with st.sidebar:
      st.subheader("Your Docs")
      pdf_docs = st.file_uploader("Upload your Docs",accept_multiple_files=True)  #type=["pdf", "txt"], accept_multiple_files=True)
      if st.button("Process"):
         #get the pdf text
         # get the text chunks
         #create the vector store
         with st.spinner("Processing..."):
            #get the pdf text
            raw_text = getPdfText(pdf_docs)
            # st.write(raw_text)

            #get the text chunks
            textChunks = getTextChunks(raw_text)

            # create the vector store
            vectorStore = getVectorStore(textChunks)
           
           #create conversAtion chain
            # conversation = getConversationChain(vectorStore)
       

   



if __name__ == '__main__':
   main()