import psycopg2
from sentence_transformers import SentenceTransformer
import time
import numpy as np

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

conn = psycopg2.connect(
    host="localhost",
    database="bookcorpus",
    user="postgres",
    password="postgres"
)

cursor = conn.cursor()
cursor.execute('SELECT * FROM sentences;')
sentences = cursor.fetchall()

update_times = []

for sentence in sentences:
    sentence_id = sentence[0]
    sentence_text = sentence[1]

    start_time = time.time()
    embedding = model.encode(sentence_text).tolist()
    cursor.execute('UPDATE sentences SET embedding = %s WHERE id = %s;', (embedding, sentence_id))
    end_time = time.time()
    update_times.append(end_time - start_time)

conn.commit()
cursor.close()
conn.close()

min_time = np.min(update_times)
max_time = np.max(update_times)
mean_time = np.mean(update_times)
std_dev_time = np.std(update_times)

print(f"Tiempo mínimo de actualización: {min_time:.6f} segundos")
print(f"Tiempo máximo de actualización: {max_time:.6f} segundos")
print(f"Tiempo medio de actualización: {mean_time:.6f} segundos")
print(f"Desviación estándar del tiempo de actualización: {std_dev_time:.6f} segundos")
