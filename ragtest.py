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

user_url = input("Enter URL Here for chatting with the webpage : ")

file_name = "web_scrapped.txt"
scrapping_request = requests.get(user_url)
soup = BeautifulSoup(scrapping_request.content, "html.parser")
scrapped_text = soup.get_text()
with open(file_name,"w",encoding="utf-8") as f:
    f.write(scrapped_text)

print("Web scrapping Complete")
print("URL : ", user_url)
print("Scrapped file : ",file_name)

loader = TextLoader(file_name)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

print("texts splitting done : " ,texts)

get_embedding_model_specs(credentials.get('url'))

embeddings = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id
    )

print("Embeddings : ",embeddings)

docsearch = Chroma.from_documents(texts, embeddings)

print("docsearch : ", docsearch)

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

qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())

query = input("Question you want to ask the webpage : ")
res_ans = qa.invoke(query)
print(res_ans)