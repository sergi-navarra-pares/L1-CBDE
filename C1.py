import chromadb
from sentence_transformers import SentenceTransformer
import time
import numpy as np

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

chroma_client = chromadb.HttpClient(host='localhost', port=8000)

euclidean_collection = chroma_client.get_collection("euc_sentences")
manhattan_collection = chroma_client.get_collection("man_sentences")

euc_documents = euclidean_collection.get()
sentences = euc_documents["documents"]
ids = euc_documents["ids"]

update_times = []

for idx, sentence in enumerate(sentences):
    print(idx)
    start_time = time.time()
    embedding = model.encode(sentence).tolist()
    euclidean_collection.update(
        ids=[ids[idx]],
        embeddings=[embedding]
    )
    end_time = time.time()
    manhattan_collection.update(
        ids=[ids[idx]],
        embeddings=[embedding]
    )
    update_times.append(end_time - start_time)

min_time = np.min(update_times)
max_time = np.max(update_times)
mean_time = np.mean(update_times)
std_dev_time = np.std(update_times)

print(f"Tiempo mínimo de actualización: {min_time:.6f} segundos")
print(f"Tiempo máximo de actualización: {max_time:.6f} segundos")
print(f"Tiempo medio de actualización: {mean_time:.6f} segundos")
print(f"Desviación estándar del tiempo de actualización: {std_dev_time:.6f} segundos")