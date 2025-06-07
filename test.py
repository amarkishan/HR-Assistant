from flask import Flask, request, jsonify
import os
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
metadata = []

folder_path = "C:/Users/18475/Documents/Chatbot"

for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)

    if filename.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        chunks = text.split("\n\n")
        embeddings = model.encode(chunks)
        index.add(np.array(embeddings).astype("float32"))
        metadata.extend([{"filename": filename, "text": chunk} for chunk in chunks])

    elif filename.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        if isinstance(json_data, list):
            texts = [entry.get("content", "") for entry in json_data if isinstance(entry, dict)]
        else:
            texts = [str(json_data)]

        embeddings = model.encode(texts)
        index.add(np.array(embeddings).astype("float32"))
        metadata.extend([{"filename": filename, "text": text} for text in texts])

faiss.write_index(index, "faiss_index.index")
np.save("faiss_metadata.npy", metadata)

# Load saved metadata and index
index = faiss.read_index("faiss_index.index")
metadata = np.load("faiss_metadata.npy", allow_pickle=True)

@app.route("/", methods=["POST"])
def get_answer():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    query_embedding = model.encode([user_query]).astype("float32")
    k = 5
    distances, indices = index.search(query_embedding, k)
    retrieved_context = "\n".join([metadata[idx]['text'] for idx in indices[0]])

    API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"
    headers = {
        "Authorization": "Bearer <your_token>",   # <-- Insert your Hugging Face API token
        "Content-Type": "application/json"
    }

    prompt = f"""You are a helpful HR assistant.\n\nContext:\n{retrieved_context}\n\nQuestion:\n{user_query}\n\nAnswer:"""

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        response_text = response.json()["choices"][0]["message"]["content"]
        cleaned_response = response_text.replace("\n\n1", " 1")
        return cleaned_response
        # return jsonify({"response": response.json()["choices"][0]["message"]["content"]})
    else:
        return jsonify({"error": response.text}), response.status_code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
