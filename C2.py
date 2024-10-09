import chromadb
import time
import numpy as np

chroma_client = chromadb.HttpClient(host='localhost', port=8000)

euclidean_collection = chroma_client.get_collection("euc_sentences")
cosine_collection = chroma_client.get_collection("cos_sentences")

ids = [str(i + 1) for i in range(10)]
selected_sentences = euclidean_collection.get(ids=ids)['documents']

euclidean_similarities_time = []
cosine_similarities_time = []

for i, sentence in enumerate(selected_sentences):
    start_time = time.time()
    euc_result = euclidean_collection.query(
        query_texts=[sentence],
        n_results=3
    )
    end_time = time.time()
    euclidean_similarities_time.append(end_time - start_time)

    start_time = time.time()
    cos_result = cosine_collection.query(
        query_texts=[sentence],
        n_results=3
    )
    end_time = time.time()
    cosine_similarities_time.append(end_time - start_time)

    print(f"""
    Sentence {i + 1}: 
        - id: {i + 1}, text: {sentence}
    Top two most similar using euclidean distance
        - id: {euc_result['ids'][0][1]}, text: {euc_result['documents'][0][1]}
        - id: {euc_result['ids'][0][2]}, text: {euc_result['documents'][0][2]}
    Top two most similar using euclidean distance
        - id: {cos_result['ids'][0][1]}, text: {cos_result['documents'][0][1]}
        - id: {cos_result['ids'][0][2]}, text: {cos_result['documents'][0][2]}
    """)

euc_min_time = np.min(euclidean_similarities_time)
euc_max_time = np.max(euclidean_similarities_time)
euc_mean_time = np.mean(euclidean_similarities_time)
euc_std_dev_time = np.std(euclidean_similarities_time)

print("Tiempos metrica de distancia euclidiana")
print(f"Tiempo mínimo de cálculo de similitudes: {euc_min_time:.6f} segundos")
print(f"Tiempo máximo de cálculo de similitudes: {euc_max_time:.6f} segundos")
print(f"Tiempo medio de cálculo de similitudes: {euc_mean_time:.6f} segundos")
print(f"Desviación estándar del tiempo de cálculo de similitudes: {euc_std_dev_time:.6f} segundos")


cos_min_time = np.min(cosine_similarities_time)
cos_max_time = np.max(cosine_similarities_time)
cos_mean_time = np.mean(cosine_similarities_time)
cos_std_dev_time = np.std(cosine_similarities_time)

print("Tiempos metrica de distancia cosine")
print(f"Tiempo mínimo de cálculo de similitudes: {cos_min_time:.6f} segundos")
print(f"Tiempo máximo de cálculo de similitudes: {cos_max_time:.6f} segundos")
print(f"Tiempo medio de cálculo de similitudes: {cos_mean_time:.6f} segundos")
print(f"Desviación estándar del tiempo de cálculo de similitudes: {cos_std_dev_time:.6f} segundos")
