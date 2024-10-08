import psycopg2
import time
import numpy as np

with open("sentences.txt", "r") as file:
    sentences = [line.strip() for line in file.readlines()]

conn = psycopg2.connect(
    host="localhost",
    database="bookcorpus",
    user="postgres",
    password="Snavarra2"
)

cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS sentences (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding FLOAT8[]
);
''')

insertion_times = []

for sentence in sentences:
    start_time = time.time()
    cursor.execute('INSERT INTO sentences (text) VALUES (%s)', (sentence,))
    end_time = time.time()
    insertion_times.append(end_time - start_time)

conn.commit()
cursor.close()
conn.close()

min_time = np.min(insertion_times)
max_time = np.max(insertion_times)
mean_time = np.mean(insertion_times)
std_dev_time = np.std(insertion_times)

print(f"Tiempo mínimo de inserción: {min_time:.6f} segundos")
print(f"Tiempo máximo de inserción: {max_time:.6f} segundos")
print(f"Tiempo medio de inserción: {mean_time:.6f} segundos")
print(f"Desviación estándar del tiempo de inserción: {std_dev_time:.6f} segundos")