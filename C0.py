import chromadb
import time
import numpy as np


chroma_client = chromadb.PersistentClient(path="./chromadb")

euclidean_collection = chroma_client.get_or_create_collection(name="euc_sentences", metadata={"hnsw:space": "l2"})
cosine_collection = chroma_client.get_or_create_collection(name="cos_sentences", metadata={"hnsw:space": "cosine"})

with open("sentences.txt", "r") as file:
    sentences = [line.strip() for line in file.readlines()]

insertion_times = []

for i, sentence in enumerate(sentences):
    print(i)
    start_time = time.time()
    euclidean_collection.add(
        documents=[sentence],
        ids=[str(i + 1)]
    )
    end_time = time.time()
    cosine_collection.add(
        documents=[sentence],
        ids=[str(i + 1)]
    )
    insertion_times.append(end_time - start_time)

min_time = np.min(insertion_times)
max_time = np.max(insertion_times)
mean_time = np.mean(insertion_times)
std_dev_time = np.std(insertion_times)

print(f"Tiempo mínimo de inserción: {min_time:.6f} segundos")
print(f"Tiempo máximo de inserción: {max_time:.6f} segundos")
print(f"Tiempo medio de inserción: {mean_time:.6f} segundos")
print(f"Desviación estándar del tiempo de inserción: {std_dev_time:.6f} segundos")
