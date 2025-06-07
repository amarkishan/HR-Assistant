# HR Assistant Chatbot API

This project implements an HR assistant chatbot API using Flask, FAISS, Sentence Transformers, and Hugging Face's Mistral model. The chatbot is designed to answer HR-related queries by retrieving relevant context from preloaded documents and generating responses using a language model.

## Features

- **Contextual Search**: Uses FAISS for efficient nearest-neighbor search to retrieve relevant context from HR documents.
- **Natural Language Understanding**: Employs Sentence Transformers for embedding user queries and document chunks.
- **AI-Powered Responses**: Integrates Hugging Face's Mistral model to generate human-like answers based on retrieved context.
- **REST API**: Provides a POST endpoint for querying the chatbot.

## Technologies Used

- **Flask**: A lightweight Python web framework for building the REST API.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **Sentence Transformers**: A pre-trained model for generating embeddings from text.
- **Hugging Face API**: Mistral-7B-Instruct model for generating responses.
- **NumPy**: For handling embeddings and metadata.
- **Requests**: For making HTTP requests to the Hugging Face API.

## Project Structure

- **`five.py`**: Main application file containing the Flask API and chatbot logic.
- **FAISS Index**: Stores embeddings for efficient retrieval of relevant document chunks.
- **Metadata**: Contains information about the source and content of document chunks.

## How It Works

1. **Document Preprocessing**:
   - Text files and JSON files are loaded from the specified folder.
   - Each document is split into chunks, and embeddings are generated using Sentence Transformers.
   - FAISS index is created for efficient similarity search, and metadata is saved for context retrieval.

2. **Query Handling**:
   - User queries are received via the POST endpoint.
   - The query is embedded using Sentence Transformers and searched against the FAISS index.
   - Relevant document chunks are retrieved based on similarity.

3. **Response Generation**:
   - The retrieved context is sent to Hugging Face's Mistral model along with the user query.
   - The model generates a response, which is returned to the user.

## API Endpoint

### `POST /`
- **Request Body**:
  ```json
  {
    "query": "Your HR-related question here"
  }
  ```
- **Response**:
  - Success:
    ```json
    {
      "response": "Generated answer based on your query"
    }
    ```
  - Error:
    ```json
    {
      "error": "Error message"
    }
    ```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install flask numpy faiss-cpu sentence-transformers requests
   ```

2. **Run the Application**:
   ```bash
   python five.py
   ```

3. **Send Queries**:
   Use tools like Postman or cURL to send POST requests to the API.

## Folder Structure

- **`Chatbot/`**: Contains HR-related documents (e.g., leave policy, payroll policy, benefits).
- **FAISS Index**: Stored as `faiss_index.index`.
- **Metadata**: Stored as `faiss_metadata.npy`.

## Future Enhancements

- Add authentication for API requests.
- Expand document coverage for more HR topics.
- Improve response generation with fine-tuned models.

## License

This project is licensed under the MIT License.