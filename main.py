
import json
import requests
import numpy as np
import faiss

from scipy.stats import spearmanr


# Load the dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = data["sentences"]
queries = data["queries"]

if len(sentences) != 50 or len(queries) != 10:
    raise ValueError("Dataset must contain exactly 50 sentences and 10 queries.")


# Generate embeddings using Ollama
def get_embeddings(model_name, text_list):
    embeddings = []

    for text in text_list:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": model_name,
                "prompt": text
            },
            timeout=180
        )

        response.raise_for_status()
        result = response.json()

        embeddings.append(result["embedding"])

    arr = np.array(embeddings, dtype=np.float32)

    # Normalize embeddings to unit length
    faiss.normalize_L2(arr)

    return arr


# Embed the 50 corpus sentences using both models
print("Embedding corpus with nomic-embed-text...")
emb_a = get_embeddings("nomic-embed-text", sentences)

print("Embedding corpus with mxbai-embed-large...")
emb_b = get_embeddings("mxbai-embed-large", sentences)


# Build separate FAISS indexes
index_a = faiss.IndexFlatIP(emb_a.shape[1])
index_a.add(emb_a)

index_b = faiss.IndexFlatIP(emb_b.shape[1])
index_b.add(emb_b)


# Embed all 10 queries using both models
print("Embedding queries with nomic-embed-text...")
emb_q_a = get_embeddings("nomic-embed-text", queries)

print("Embedding queries with mxbai-embed-large...")
emb_q_b = get_embeddings("mxbai-embed-large", queries)


# Retrieve rankings and calculate Spearman correlation
matrix = []

for q in range(len(queries)):

    print(f"Processing query {q + 1}/10...")

    q_vec_a = emb_q_a[q:q + 1]
    q_vec_b = emb_q_b[q:q + 1]

    # Retrieve all 50 documents in ranked order
    D_a, I_a = index_a.search(q_vec_a, 50)
    D_b, I_b = index_b.search(q_vec_b, 50)

    # Top three document indices
    top3_a = I_a[0][:3].tolist()
    top3_b = I_b[0][:3].tolist()

    # Align similarity scores by original document index
    scores_a = np.zeros(50)
    scores_b = np.zeros(50)

    scores_a[I_a[0]] = D_a[0]
    scores_b[I_b[0]] = D_b[0]

    # Calculate Spearman rank correlation
    rho, _ = spearmanr(scores_a, scores_b)

    row = [
        int(top3_a[0]),
        int(top3_a[1]),
        int(top3_a[2]),
        int(top3_b[0]),
        int(top3_b[1]),
        int(top3_b[2]),
        round(float(rho), 4)
    ]

    matrix.append(row)


# Print final JSON matrix
output = json.dumps(matrix)

print("\nFINAL 10 x 7 MATRIX:")
print(output)

# Save results to a file
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(matrix, f, indent=2)

print("\nResults saved to results.json")
