# Zoom Citra

Kelompok 3 - R8E

Aplikasi webcam interaktif untuk demonstrasi teknik pengolahan citra real-time menggunakan OpenCV dan MediaPipe.

## Fitur

- Zoom kamera dengan gesture dua tangan.
- Filter pengolahan citra real-time:
  - Normal
  - Grayscale
  - Negative
- Simpan hasil frame ke folder `outputs`.
- UI overlay berisi status filter, zoom, dan gesture.

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Program

```bash
python zoom_inout.py
```

Saat pertama dijalankan, program akan mengunduh model MediaPipe `hand_landmarker.task`.

## Troubleshooting

Jika muncul error `ModuleNotFoundError: No module named 'mediapipe'`, jalankan:

```bash
python -m pip install -r requirements.txt
```

Jika `mediapipe` gagal diinstall, cek versi Python:

```bash
python --version
```

Gunakan Python 3.9 sampai 3.12 versi 64-bit agar kompatibel dengan package `mediapipe`.

## Kontrol

| Tombol | Fungsi |
| --- | --- |
| `1` | Filter Normal |
| `2` | Filter Grayscale |
| `3` | Filter Negative |
| `S` | Menyimpan frame hasil |
| `R` | Reset zoom |
| `Q` | Keluar |
