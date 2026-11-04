import pandas as pd

RAW_PATH = "data/scrape_kompas_all.xlsx"
CLEAN_PATH = "data/scrape_kompas_clean.csv"

print("1. Loading raw data...")
df_raw = pd.read_excel(RAW_PATH)
print(f"Shape Raw: {df_raw.shape}")

print("\n2. Loading clean data...")
df_clean = pd.read_csv(CLEAN_PATH)
print(f"Shape Clean: {df_clean.shape}")

print("-" * 50)
print(f"Selisih baris yang dibuang pembuat dataset: {len(df_raw) - len(df_clean)} baris!")

print("\n3. Cek duplikasi di data mentah:")
exact_dups = df_raw.duplicated().sum()
print(f"duplikat persis (seluruh kolom identik): {exact_dups} baris")

link_dups = df_raw.duplicated(subset=["link"]).sum()
print(f"duplikat berdasarkan link (artikel yang sama muncul di keyword yang beda) {link_dups}")

print("\n4. Contoh artikel yang muncul di banyak keyword:")
sample_dup_link = df_raw[df_raw.duplicated(subset=["link"], keep=False)]["link"].iloc[0]
dup_samples = df_raw[df_raw["link"] == sample_dup_link][['link', 'title', 'keywords', 'category']]
print(dup_samples)