import chromadb
import time
import numpy as np

chroma_client = chromadb.HttpClient(host='localhost', port=8000)

euclidean_collection = chroma_client.get_collection("euc_sentences")
manhattan_collection = chroma_client.get_collection("man_sentences")

ids = [str(i + 1) for i in range(10)]
selected_sentences = euclidean_collection.get(ids=ids)['documents']
euclidean_collection.delete(ids=ids)
manhattan_collection.delete(ids=ids)

euclidean_similarities_time = []
manhattan_similarities_time = []

for i, sentence in enumerate(selected_sentences):
    start_time = time.time()
    euc_result = euclidean_collection.query(
        query_texts=[sentence],
        n_results=2
    )
    end_time = time.time()
    euclidean_similarities_time.append(end_time - start_time)

    start_time = time.time()
    man_result = manhattan_collection.query(
        query_texts=[sentence],
        n_results=2
    )
    end_time = time.time()
    manhattan_similarities_time.append(end_time - start_time)

    print(f"""
    Sentence {i + 1}: 
        - id: {i + 1}, text: {sentence}
    Top two most similar using euclidean distance
        - id: {euc_result['ids'][0][0]}, text: {euc_result['documents'][0][0]}
        - id: {euc_result['ids'][0][1]}, text: {euc_result['documents'][0][1]}
    Top two most similar using euclidean distance
        - id: {man_result['ids'][0][0]}, text: {man_result['documents'][0][0]}
        - id: {man_result['ids'][0][1]}, text: {man_result['documents'][0][1]}
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


man_min_time = np.min(manhattan_similarities_time)
man_max_time = np.max(manhattan_similarities_time)
man_mean_time = np.mean(manhattan_similarities_time)
man_std_dev_time = np.std(manhattan_similarities_time)

print("Tiempos metrica de distancia manhattan")
print(f"Tiempo mínimo de cálculo de similitudes: {man_min_time:.6f} segundos")
print(f"Tiempo máximo de cálculo de similitudes: {man_max_time:.6f} segundos")
print(f"Tiempo medio de cálculo de similitudes: {man_mean_time:.6f} segundos")
print(f"Desviación estándar del tiempo de cálculo de similitudes: {euc_std_dev_time:.6f} segundos")
