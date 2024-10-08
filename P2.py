import psycopg2
import numpy as np
import time
import numpy as np

conn = psycopg2.connect(
    host="localhost",
    database="bookcorpus",
    user="postgres",
    password="Snavarra2"
)


def calculate_top_2_euclidian(selected_sentence, sentence_list):
    selected_sentence_embedding = selected_sentence[2]

    eucliean_distances = []
    for sentence in sentence_list:
        sentence_id, sentence_text, sentence_embedding = sentence
        eucliean_distance = np.linalg.norm(np.array(selected_sentence_embedding) - np.array(sentence_embedding))
        eucliean_distances.append((sentence_id, sentence_text, eucliean_distance))

    euclidean_sorted_distances = sorted(eucliean_distances, key=lambda x: x[2])
    return euclidean_sorted_distances[:2]


def calculate_top_2_manhattan(selected_sentence, sentence_list):
    selected_sentence_embedding = selected_sentence[2]

    manhattan_distances = []
    for sentence in sentence_list:
        sentence_id, sentence_text, sentence_embedding = sentence
        manhattan_distance = np.sum(np.abs(np.array(selected_sentence_embedding) - np.array(sentence_embedding)))
        manhattan_distances.append((sentence_id, sentence_text, manhattan_distance))

    manhattan_sorted_distances = sorted(manhattan_distances, key=lambda x: x[2])
    return manhattan_sorted_distances[:2]


cursor = conn.cursor()

cursor.execute("SELECT * FROM sentences ORDER BY id;")
all_sentences = cursor.fetchall()

cursor.close()
conn.close()

selected_sentences = all_sentences[:10]
other_sentences = all_sentences[10:]

euclidean_similarities_time = []
manhattan_similarities_time = []

for i, selected_sentence in enumerate(selected_sentences):
    start_time = time.time()
    euclidean_top_similarities = calculate_top_2_euclidian(selected_sentence, other_sentences)
    end_time = time.time()
    euclidean_similarities_time.append(end_time - start_time)

    start_time = time.time()
    manhattan_top_similarities = calculate_top_2_manhattan(selected_sentence, other_sentences)
    end_time = time.time()
    manhattan_similarities_time.append(end_time - start_time)
    
    
    print(f"""
    Sentence {i + 1}: 
        - id: {selected_sentence[0]}, text: {selected_sentence[1]}
    Top two most similar using euclidean distance:
        - id: {euclidean_top_similarities[0][0]}, text: {euclidean_top_similarities[0][1]}
        - id: {euclidean_top_similarities[1][0]}, text: {euclidean_top_similarities[1][1]}
    Top two most similar using manhattan distance:
        - id: {manhattan_top_similarities[0][0]}, text: {manhattan_top_similarities[0][1]}
        - id: {manhattan_top_similarities[1][0]}, text: {manhattan_top_similarities[1][1]}
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

