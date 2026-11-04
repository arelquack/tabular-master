import pandas as pd

df = pd.read_csv("data/scrape_kompas_clean.csv")

print("=== CONTOH ARTIKEL 1 ===")
print("JUDUL:", df['title'].iloc[0])
print("\nISI TEKS (500 karakter pertama):")
print(df['content'].iloc[0][:500])

print("\nISI TEKS (500 karakter terakhir):")
print(df['content'].iloc[0][-500:])

print("\n" + "="*50 + "\n")

print("=== CONTOH ARTIKEL 2 ===")
print("JUDUL:", df['title'].iloc[5])
print("\nISI TEKS(500 karakter terakhir):")
print(df['content'].iloc[5][-500:])