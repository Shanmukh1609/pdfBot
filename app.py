import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

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
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_texts(textChunks, embeddings)
    return vectorStore

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
           
       

   



if __name__ == '__main__':
   main()