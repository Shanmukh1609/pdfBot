# pdfBot

## Concepts Utilized 
Semantic Searching, Vector Database, Embedding, Memory of Conversations, QA model regarding the pdfs.

## Description

## Working
- A PDF Bot that is capable of taking multiple PDFs as the input.
- It converts the contents of pdfs into chunks of text.
-  ``` gemini-embedding-001 ``` is used to generate embeddings of the text.
-  Each embedding is of default size 3072 and it can be adjusted to 768, 1536.
-  Embeddings are stored in the   ``` FAISS ``` vector db.
-  When a query is asked by the user, it searches for the semantically similar embeddings to user request query.
-  The embeddings of context and query are given to ``` gemini-2.5 pro ``` model to generate context aware answers.

## SETUP
- Visit [setup.md](./setup.md) to run the project.


  
  
