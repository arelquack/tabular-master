import os
import glob

def main():
    print("======================================================================")
    print("  MEMERIKSA KELENGKAPAN OFFLINE BUNDLE...")
    print("======================================================================")
    wheels = glob.glob('wheels/*.whl')
    models_dir = 'models'
    models = []
    if os.path.exists(models_dir):
        models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and os.path.exists(os.path.join(models_dir, d, "config.json"))]
    
    print(f"Total Binary Wheels (.whl) Terunduh : {len(wheels)}")
    print(f"Total Model Offline Terverifikasi   : {len(models)}")
    for m in models:
        print(f"  - ./models/{m}")
    
    if len(wheels) > 0 and len(models) >= 3:
        print("\n>>> STATUS: 100% SUKSES! BUNDLE SIAP DISALIN KE FLASHDISK! <<<")
        print("Petunjuk ke PC Panitia:")
        print("  1. Salin seluruh isi folder ini ke Flashdisk.")
        print("  2. Di PC Panitia, jalankan install_offline.bat (atau install_offline.ps1)")
        print("  3. Jalankan launch_studio.bat (atau launch_studio.ps1) untuk membuka platform.")
    else:
        print("\n[PERINGATAN] Ada file yang belum lengkap.")
        if len(wheels) == 0:
            print("  - Folder wheels/ masih kosong. Pastikan pip download berjalan sukses.")
        if len(models) < 3:
            print(f"  - Model offline baru ada {len(models)} model terverifikasi.")
        print("Silakan jalankan ulang skrip packaging jika koneksi internet sempat terputus.")
    print("======================================================================\n")

if __name__ == "__main__":
    main()
