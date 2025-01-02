import csv

# Membuka file CSV
with open('bikes.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)  # Menampilkan setiap baris sebagai daftar