# 🛵 Food Delivery Cost Prediction — End-to-End ML Project

> Menganalisis dan memprediksi biaya pengiriman makanan online menggunakan pendekatan data-driven, dari eksplorasi mendalam hingga deployment model machine learning.

**Muhammad Binar Raffi Lazuardi** · Statistika 2023

---

## 🧠 Pendekatan & Cara Berpikir

Proyek ini bukan sekadar membangun model — saya memulai dari **pertanyaan bisnis yang nyata**: *apa yang sebenarnya membuat biaya pengiriman naik atau turun?* Dari sana, setiap keputusan analisis punya alasan yang jelas.

Alur berpikir saya:

```
Pertanyaan Bisnis → Pahami Data → Bersihkan dengan Bijak → Gali Insight → Bangun Model → Interpretasi
```

---

## 📌 Problem Statement

Industri pengiriman makanan online menghadapi **variasi harga pengiriman yang tidak konsisten** — dipengaruhi jarak, cuaca, hingga lonjakan permintaan. Dua pertanyaan utama yang saya jawab:

1. **Faktor apa saja yang benar-benar memengaruhi biaya pengiriman?**
2. **Kondisi seperti apa yang paling optimal bagi perusahaan?**

---

## 📂 Dataset

- **Ukuran:** 20.355 baris × 12 kolom
- **Periode:** 27 November – 13 Desember 2023
- **Target:** `biaya_pengiriman` (Rupiah)

| Kolom | Tipe | Keterangan |
|---|---|---|
| `waktu_pemesanan` | datetime | Timestamp pemesanan |
| `biaya_pengiriman` | float | **Target variabel** |
| `aktivitas_aplikasi` | int | Jumlah aktivitas user di app |
| `kategori_makanan` | category | Jenis makanan (nominal) |
| `jarak_terjauh` | float | Jarak pengiriman terjauh |
| `jarak_rata2` | float | Rata-rata jarak pengiriman |
| `jarak_terdekat` | float | Jarak pengiriman terdekat |
| `hujan` | float | Indikator curah hujan |
| `kelembapan_udara` | float | Tingkat kelembapan |
| `suhu_udara` | float | Suhu udara |
| `kecepatan_angin` | float | Kecepatan angin |
| `kondisi_awan` | float | Persentase tutupan awan |

---

## 🔍 Analytical Thinking — Highlight Keputusan Penting

### 1. EDA Sebelum Cleaning, Bukan Sebaliknya
Saya memilih melakukan EDA *setelah* cek missing value tapi *sebelum* dropping outlier — karena saya ingin melihat kondisi data asli sebelum distribusinya berubah. Ini penting agar keputusan pembersihan data tidak dilakukan secara membabi buta.

### 2. Deteksi Anomali Time Series
Dari visualisasi scatter plot waktu vs. biaya pengiriman, saya menemukan **gap data pada 5–12 Desember 2023**. Alih-alih langsung drop, saya investigasi dulu: apakah karena hujan? libur? atau memang data hilang?

```
Hipotesis → Cek variabel hujan → Cek kelembapan → Konfirmasi: data hilang, bukan fenomena nyata
```

Gap ini kemudian ditangani dengan **interpolasi linear + rolling mean** agar time series tetap kontinu.

### 3. Transformasi Variabel Berdasarkan Distribusi
Tidak semua outlier diperlakukan sama:

| Variabel | Masalah | Solusi |
|---|---|---|
| `aktivitas_aplikasi` | Right-skewed + outlier | IQR removal → distribusi normal sendiri |
| `jarak_terdekat` | Right-skewed + outlier | IQR removal + **log transform** |
| `suhu_udara` | Outlier persisten | IQR removal + **Box-Cox transform** |
| `kenaikan_tarif_minimum` | Data sangat imbalanced | **Drop kolom** — noise, bukan signal |

### 4. Cuaca Tidak Linear → Pakai PCA
Heatmap korelasi menunjukkan **variabel cuaca tidak memiliki hubungan linear** dengan `biaya_pengiriman`. Daripada abaikan, saya gabungkan 5 variabel cuaca menjadi satu komponen `combined_weather` menggunakan **PCA** — mengurangi dimensi tanpa kehilangan informasi penting.

### 5. Segmentasi Pelanggan dengan K-Means
Scatter plot `aktivitas_aplikasi` vs `biaya_pengiriman` terlihat acak. Saya pakai **Elbow Method** untuk menentukan k optimal (k=3), lalu K-Means untuk menemukan 3 segmen tersembunyi:

| Klaster | Profil | Rekomendasi Bisnis |
|---|---|---|
| 0 | Aktivitas rendah, biaya rendah | Berikan promo & diskon |
| 1 | Aktivitas sedang, biaya tinggi | Buat loyalty membership |
| 2 | Aktivitas tinggi, biaya sangat tinggi | Exclusive membership premium |

### 6. Time Series Feature — Fail Fast
Saya mencoba menambahkan fitur temporal (jam, hari, bulan dengan **cyclic encoding** sin/cos) dan evaluasi dengan **TimeSeriesSplit**. Hasilnya: tidak ada peningkatan performa. Keputusan: drop `waktu_pemesanan`. *Fail fast, move on.*

---

## 🤖 Modelling

**Metrik evaluasi:** RMSE — agar prediksi yang jauh dari nilai aktual lebih "dihukum"

Saya membandingkan 5 model sekaligus dalam satu pipeline:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | lebih tinggi | lebih tinggi | lebih rendah |
| Gradient Boosting | lebih tinggi | lebih tinggi | lebih rendah |
| XGBoost | lebih tinggi | lebih tinggi | lebih rendah |
| LightGBM | lebih tinggi | lebih tinggi | lebih rendah |
| **CatBoost ✅** | **792.40** | **573.06** | **0.992** |

**CatBoost** menang di semua metrik. Model mampu menjelaskan **99.2% variabilitas** data target — sangat baik untuk data dengan banyak pola non-linear.

> Hyperparameter tuning via RandomizedSearchCV juga dicoba, namun tidak memberikan peningkatan signifikan → base model sudah optimal.

---

## 🏆 Feature Importance — Apa yang Paling Berpengaruh?

```
1. kategori_makanan                              ████████████████████ paling dominan
2. cluster_aktivitas_aplikasi_biaya_pengiriman   ███████████████
3. cluster_combined_weather_biaya_pengiriman     ████████████
4. jarak_terdekat                                ██████████
5. jarak_rata2                                   ████████
```

**Interpretasi:**
- `kategori_makanan` dominan karena jenis makanan memengaruhi kebutuhan logistik (pengemasan, berat, handling khusus)
- Fitur hasil clustering (`cluster_*`) terbukti *lebih informatif* dibanding variabel mentahnya — ini validasi bahwa pendekatan segmentasi tadi tepat
- Variabel cuaca berkontribusi kecil tapi tetap ada, artinya keputusan untuk tidak membuangnya benar

---

## 💡 Business Insight

Kondisi ideal untuk perusahaan memaksimalkan efisiensi & pendapatan:

- **Fokus pada segmen Klaster 2** (high activity, high spend) — kontributor pendapatan terbesar
- **Kelompokkan menu berdasarkan kebutuhan logistik**, bukan sekadar kategori kuliner
- **Prioritaskan pengiriman saat cuaca stabil** (Klaster cuaca 0 & 1); siapkan mitigasi saat ekstrem
- **Optimalkan jarak terdekat** — kategori makanan populer cenderung dekat, ini keuntungan kompetitif yang bisa diperkuat

---

## 🛠️ Tech Stack

```python
pandas · numpy · scikit-learn · seaborn · matplotlib
xgboost · lightgbm · catboost · scipy · joblib
```

---

## 📁 Struktur Project

```
📁 project/
├── POJOKSTATISTIK_Binar.ipynb   # Main notebook
├── best_model.pkl               # Saved CatBoost model
└── README.md
```

---

## ▶️ Cara Menjalankan

```bash
pip install catboost lightgbm xgboost scikit-learn pandas numpy seaborn matplotlib scipy
```

Jalankan `POJOKSTATISTIK_Binar.ipynb` dari atas ke bawah. Dataset otomatis diunduh dari Google Drive saat cell *Loading Data* dieksekusi.
