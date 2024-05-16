# RAG Streamlit Application with IBM Model

This project is a Streamlit application that performs Retrieval-Augmented Generation (RAG) using IBM Watson models. The application scrapes text from a user-provided URL, processes the text, stores it in a Chroma vector database, and then uses IBM's Watson models to answer user queries based on the scraped content.

## Requirements

- Python 3.8+
- Streamlit
- Requests
- BeautifulSoup4
- python-dotenv
- langchain
- langchain_ibm
- ibm-watson-machine-learning

## Setup

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory of the project and add your IBM Cloud credentials:
    ```plaintext
    API_KEY=your_ibm_cloud_api_key
    PROJECT_ID=your_ibm_project_id
    ```

## Running the Application

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open the application in your web browser. The default URL is usually `http://localhost:8501`.

## Usage

1. Enter a URL in the input box to scrape the webpage.
2. The application will scrape the text from the provided URL and process it in the following steps:
    - Scrape the webpage text and save it to a file.
    - Split the text into chunks using `CharacterTextSplitter`.
    - Generate embeddings for the text chunks using IBM's Watson embeddings.
    - Store the embeddings in a Chroma vector database.
3. Once the text is processed, enter a query to ask a question based on the scraped content.
4. The application will use IBM's Granite models to generate a response based on the stored content.

## Code Overview

- **Streamlit UI**: 
  - The main input is for the URL to be scraped.
  - An additional input allows users to enter queries once the text is processed.

- **Web Scraping**:
  - Uses `requests` to fetch the webpage content.
  - `BeautifulSoup` extracts text from the HTML.

- **Text Processing**:
  - `TextLoader` loads the scraped text.
  - `CharacterTextSplitter` splits the text into manageable chunks.

- **Embedding and Storage**:
  - `WatsonxEmbeddings` generates embeddings for the text chunks.
  - `Chroma` stores the embeddings in a vector database.

- **RAG Model**:
  - Initializes IBM's WatsonxLLM with specified model parameters.
  - Uses `RetrievalQA` to create a retrieval-augmented generation pipeline.
  - Processes user queries against the stored embeddings and generates answers.

## Note

Ensure your IBM Cloud credentials are valid and have the necessary permissions to access the Watson services.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize and enhance this application as needed. Contributions are welcome!

---

**Disclaimer**: This application is for educational purposes and is provided "as is" without warranty of any kind. The authors are not responsible for any damages or losses arising from the use of this application.
