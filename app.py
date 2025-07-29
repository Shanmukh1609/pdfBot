import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
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
       #  embeddings = OpenAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=gemini_api_key)
    vectorStore = FAISS.from_texts(texts=textChunks, embedding=embeddings)
    print("Stored in the vector store")
    return vectorStore

def getConversationChain(vectorStore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        google_api_key=gemini_api_key,
        max_output_tokens=512,
        convert_system_message_to_human=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handleUserQuery(user_query):
    response = st.session_state.conversation.invoke({"question": user_query})
    st.session_state.chat_history = response['chat_history']

    # Iterate through the chat history and display messages using templates
    for i, message in enumerate(st.session_state.chat_history):
        # LangChain message objects have a 'type' attribute (e.g., 'human', 'ai')
        # and the content is in the '.content' attribute.
        print(message)
        if message.type == 'human':
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif message.type == 'ai':
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        # Fallback if message type is unexpected, though 'human' and 'ai' are standard
        else:
            st.write(f"**{message.type.capitalize()}:** {message.content}")


def main():
    load_dotenv()   # Load environment variables from .env file
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")

    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:  #Making the variable persistent
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs: books, articles")
    user_query = st.text_input("Ask a question about your docs:")

    if user_query:
        handleUserQuery(user_query)

    # Display initial chat messages only if history is empty
    if not st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", "Hello Sai"), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", "Hello Shannu"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Docs")
        pdf_docs = st.file_uploader("Upload your Docs", accept_multiple_files=True, type=["pdf"]) #type=["pdf", "txt"], accept_multiple_files=True)
        if st.button("Process"):
         #get the pdf text
         # get the text chunks
         #create the vector store
            with st.spinner("Processing..."):
                #get the pdf text
                raw_text = getPdfText(pdf_docs)
                 # st.write(raw_text)
                 # get the text chunks
                textChunks = getTextChunks(raw_text)
                 # create the vector store
                vectorStore = getVectorStore(textChunks)
         
                st.session_state.conversation = getConversationChain(vectorStore)
                st.success("Processing complete! You can now ask questions.")
   
    return st.session_state.conversation
          #create conversAtion chain
           #takes the history of the conversation and the vector store
           #Streamlit has a tendency to reload the page
if __name__ == '__main__':
    main()
