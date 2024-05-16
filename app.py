#Streamlit application for RAG with IBM Model

import streamlit as st
import os
import requests
from dotenv import main 
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
from langchain.chains import RetrievalQA

main.load_dotenv()
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.environ["API_KEY"]
}
try:
    project_id = os.environ["PROJECT_ID"]
except KeyError:
    project_id = input("Please enter your project_id (hit enter): ")

print("Project ID : " ,project_id)

st.write("RAG Using IBM Granite Models - Chat with Web")

st.text_input("Enter URL to scrap the webpage : ",key="user_url")

if st.session_state.user_url != "":
    with st.spinner(text="Processing the URL and scrapping the text"):
        file_name = "web_scrapped.txt"
        scrapping_request = requests.get(st.session_state.user_url)
        soup = BeautifulSoup(scrapping_request.content, "html.parser")
        scrapped_text = soup.get_text()
        with open(file_name,"w",encoding="utf-8") as f:
             f.write(scrapped_text)
        print("Web scrapping Complete")
        print("URL : ", st.session_state.user_url)
        print("Scrapped file : ",file_name)

    with st.spinner(text="Breaking the text into chunks using text splitter"):
        loader = TextLoader(file_name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        print("texts splitting done : " ,texts)

    with st.spinner(text="Embedding the text chunks using WatsonEmbeddings"):
        get_embedding_model_specs(credentials.get('url'))
        embeddings = WatsonxEmbeddings(
            model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
            url=credentials["url"],
            apikey=credentials["apikey"],
            project_id=project_id
            )
        print("Embeddings : ",embeddings)

    with st.spinner(text="Storing it in Chroma vector DB"):
        docsearch = Chroma.from_documents(texts, embeddings)
        print("docsearch : ", docsearch)

    with st.spinner(text="RAG based query initialization in progress..!!"):
        model_id = ModelTypes.GRANITE_13B_CHAT_V2
        parameters = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 100,
            GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
        }
        watsonx_granite = WatsonxLLM(
            model_id=model_id.value,
            url=credentials.get("url"),
            apikey=credentials.get("apikey"),
            project_id=project_id,
            params=parameters
        )

    with st.spinner(text="Preparing the Model to launch..!!"):
        qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())
        st.text_input("Enter URL to scrap the webpage : ",key="query")
        #query = input("Question you want to ask the webpage : ")
        res_ans = qa.invoke(st.session_state.query)
        st.write(res_ans)
else:
    st.write("Enter a URL to start working")
