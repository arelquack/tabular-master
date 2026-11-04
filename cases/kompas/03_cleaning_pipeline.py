import pandas as pd
import re

def clean_compas_text(text):
    if not isinstance(text, str):
        return ""

    # 1. Hapus boilerplate 'Baca juga: ...' hingga akhir baris/tanda titik
    text = re.sub(r'Baca juga:.*?(?=\n|\.|$)', ' ', text, flags=re.IGNORECASE)

    # 2. Hapus format awal berita: misal 'JAKARTA, KOMPAS.com - '
    text = re.sub(r'^[A-Z\s]+, \s*KOMPAS\.com\s*[-–—]\s*', ' ', text)

    # 3. Hapus boilerplate ajakan unduh aplikasi / follow WhatsApp channel di akhir
    text = re.sub(r'Dapatkan update berita pilihan.*?kompas\.com.*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Simak breaking news.*?ponselmu.*', ' ', text, flags=re.IGNORECASE)

    # 4. Hapus info penulis & editor di kurung akhir: (Penulis: ... | Editor: ...)
    text = re.sub(r'\(Penulis:.*?Editor:.*?\)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Editor:.*?\)', ' ', text, flags=re.IGNORECASE)

    # 5. Hapus URL web jika ada
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 6. Hapus karakter non-alfanumerik berlebih (tapi sisakan huruf dan spasi)
    # Catatan: kita ganti tanda baca dengan spasi agar kata tidak menempel
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # 7. Hapus spasi berlebih (whitespace normalization)
    text = re.sub(r'\s+', '')