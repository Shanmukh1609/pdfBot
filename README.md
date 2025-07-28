# pdfBot

**Concepts Utilized** : Semantic Searching, Vector Database, Embedding, Memory of Conversations, QA model regarding the pdfs.

One or more pdf can be taken as the input. The pdfs are converted into chunks of text. I am using google's ``` gemini-embedding-001 ``` as the embedding model to generate the embedding for the
chunks of the text. I am not using the model directly, which means that I am using the API to get the embeddings. which raises the questions such as
- What is the limit on the chunks of text or tokens it can embed?
- What is the size of each embedding?

After generation of the embedding, we do store them in the faiss vector database. 
After that we are using the language model to answer the question. 

- What is the memory ie how it saves and uses past conversations?
- How does it use the semamtic similarity between the query and the embeddings of the database?

-Take a time of 2-3 hours. Complete the following
- Get a good understanding
- Write a good readme file
- Make the repo public
- Add it in the resume.
  
  
