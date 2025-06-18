# 🍌 🍎 🍊  SmartFruit Analyzer Pro

**Deteksi Jenis dan Kematangan Buah Otomatis Berbasis Computer Vision**

---

## 👥 Anggota Kelompok 6
- Muhammad Faisal A (152023100)
- Samuel Tambunan (152023108)
- Santana Alfandi (152023204)

Mata Kuliah: Pengolahan Citra Digital  
Semester: Genap 2024/2025  
Kelas: IFB-208

---

## 📌 Deskripsi Singkat Proyek

SmartFruit Analyzer Pro adalah aplikasi berbasis Python dan OpenCV yang dirancang untuk mendeteksi jenis buah (apel, jeruk, dan pisang) serta mengklasifikasikan tingkat kematangan pisang ke dalam 7 level berdasarkan warna kulitnya. Aplikasi ini memanfaatkan segmentasi warna HSV untuk mengenali buah secara visual dan menyertakan fitur pemrosesan citra seperti grayscale, peningkatan kontras (CLAHE), dan deteksi tepi (Canny). Dengan antarmuka GUI berbasis PyQt5, pengguna dapat memuat gambar, memilih jenis buah, dan menjalankan proses deteksi serta klasifikasi dengan mudah. Proyek ini bertujuan mendukung otomasi analisis buah secara akurat dan efisien sebagai solusi atas kendala penilaian kematangan secara manual.

SmartFruit Analyzer Pro adalah aplikasi berbasis Python dan OpenCV yang mampu:
- Mendeteksi **jenis buah** (apel, jeruk, pisang) berdasarkan segmentasi warna HSV.
- Mengklasifikasikan **tingkat kematangan pisang** ke dalam **7 level** (dari hijau hingga berbintik) berdasarkan warna kulit pisang.
- Menyediakan fitur **pengolahan citra** seperti grayscale, peningkatan kontras (CLAHE), dan deteksi tepi (Canny).
- Menampilkan hasil secara visual melalui **GUI interaktif berbasis PyQt5**.

---

## ⚙️ Fitur Aplikasi

- 🔍 **Deteksi Buah** berdasarkan warna (apel merah/hijau, jeruk, pisang)
- 📊 **Klasifikasi Kematangan Pisang** (Level 1 s.d. 7)
- 🎛️ **Preprocessing**:
  - Resize gambar
  - Enhance contrast (CLAHE)
  - Edge detection (Canny)
- 🖥️ **GUI User Interface**:
  - Load image
  - Pilih jenis buah
  - Deteksi & klasifikasi via tombol
  - Tampilkan hasil visual

---

## 🧠 Teknologi yang Digunakan

- Python 3.x
- OpenCV
- NumPy
- PyQt5



