# 🚗 Vehicle Detection Dataset Collector

Proyek ini bertujuan untuk mengumpulkan dataset gambar kendaraan dari siaran CCTV publik menggunakan model deteksi YOLOv8. Hasil deteksi divisualisasikan, dan pengguna dapat menyimpan frame tertentu dengan label yang diberikan secara manual melalui keyboard.

## 📦 Fitur

- Streaming video dari CCTV publik (real-time).
- Deteksi objek kendaraan menggunakan YOLOv8 (versi `yolov8n.pt`).
- Menampilkan bounding box dan label deteksi.
- Interaksi manual untuk menyimpan frame terdeteksi dengan label sesuai tombol keyboard.
- Penyimpanan dataset otomatis dalam folder `dataset`.

## 🖥️ Prasyarat

- Python 3.8+
- Paket Python:
  - `ultralytics`
  - `opencv-python`

Install dependensi:

bash  
`pip install ultralytics opencv-python`

## 🚀 Menjalankan Program
bash  
`python CollectDataset.py`

## 🎮 Cara Penggunaan
-Tekan `a` hingga `z` atau simbol lain untuk memberi label pada frame dan menyimpannya.  
-Tekan `s` untuk melewati frame tanpa menyimpan.  
-Tekan `q` untuk keluar dari program.  

Semua frame yang disimpan akan masuk ke folder `dataset/` dengan nama format:  
`[label]_[timestamp].jpg`  
Contoh:  
`a_1715678912.jpg`  

