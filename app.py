import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from chatTemplate import css,bot_template,user_template

# from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings

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
   #How memory works, entity memory and etc.
     # We need the memory.
     #How is data stored in the memory.(Type-In embeddings or binary or vector db)
   memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
 
   #We need the language model.
   llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512},task="text2text-generation")
   conversation_chain=ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vectorStore.as_retriever(),
      memory=memory
   )
   return conversation_chain

def handleUserQuery(user_query):
   response = st.session_state.conversation({"question": user_query})
   st.session_state.chat_history = response['chat_history']
   
   for i,message in enumerate(st.session_state.chat_history):
      if i%2 == 0:
         st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
      else:
         st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
   
def main():
   load_dotenv()  # Load environment variables from .env file
   st.set_page_config(page_title="Chat with PDFs",page_icon="books")
   
   st.write(css,unsafe_allow_html=True)
   
   if 'conversation' not in st.session_state: #Making the variable persistent
      st.session_state.conversation = None
   
   st.header("Chat with multiple PDFs: books,articles")
   user_query= st.text_input("Ask a question about your docs:")
   
   if user_query:
      handleUserQuery(user_query)
      
   st.write(user_template.replace("{{MSG}}","Hello Sai"),unsafe_allow_html= True)
   
   st.write(bot_template.replace("{{MSG}}","Hello Shannu"),unsafe_allow_html= True)
   
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
           #takes the history of the conversation and the vector store
           #Streamlit has a tendency to reload the page
            st.session_state.conversation = getConversationChain(vectorStore) #allows us to use the conversation chain in the main function
   
   return st.session_state.conversation
       

   



if __name__ == '__main__':
   main()