# PDF Chatbot

PDF chatbot is a simple application that allows you to chat with your own uploaded documents (pdfs)

## Architecture

The simple workflow the app is going to follow is represented in this diagram

![alt text](./assets/image.png)


Chatbot backend elaborated:

![alt text](./assets/workflow.png)

## Infrastructure

1. LLM - GPT/Ollama/Huggingface
2. UI - Streamlit

## Backend Logic

- Simple PDF chat application

    We upload the files -> Extract the text from the pdf -> chunk the text -> convert chunks into embeddings

    -> 

- Embedding Logic: 

    Embeddings are numbers that represent text.

    ![Embeddings](https://th.bing.com/th/id/OIP.CmsOYMVq_Eo9mddsMxe8ewHaF_?rs=1&pid=ImgDetMain)
    