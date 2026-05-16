from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load model BERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
df = pd.read_csv('dataset_final.csv')

# Ambil teks dataset
texts = df['teks'].astype(str).tolist()

# Encode dataset
dataset_embeddings = model.encode(texts)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']

    # Encode pertanyaan user
    user_embedding = model.encode([user_message])

    # Hitung similarity
    similarities = cosine_similarity(user_embedding, dataset_embeddings)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    response = texts[best_index]

    return jsonify({
        'response': response,
        'similarity': round(float(best_score) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)