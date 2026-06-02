# 🛵 Apa yang Sebenarnya Membuat Biaya Pengiriman Makanan Naik?

> Sebuah investigasi data dari 20.000+ transaksi pengiriman makanan online — mencari tahu faktor mana yang benar-benar berpengaruh, mana yang cuma noise, dan kondisi seperti apa yang ideal bagi perusahaan.

**Muhammad Binar Raffi Lazuardi** · Statistika 2023

---

## Pertanyaan yang Ingin Dijawab

Bukan sekadar "bikin model prediksi" — saya mulai dari dua pertanyaan bisnis yang konkret:

1. **Faktor apa yang benar-benar memengaruhi biaya pengiriman?**
2. **Kondisi seperti apa yang optimal bagi perusahaan?**

Dari dua pertanyaan ini, seluruh alur analisis dirancang — bukan sebaliknya.

---

## Dataset

- **20.355 transaksi** · periode 27 Nov – 13 Des 2023
- 12 variabel: waktu, jarak, cuaca, aktivitas aplikasi, kategori makanan, kenaikan tarif
- **Target:** `biaya_pengiriman` (Rupiah)

---

## 🔍 Investigasi: Menemukan Cerita di Balik Data

### Ada yang aneh di data waktu

Hal pertama yang saya lakukan adalah memplot biaya pengiriman terhadap waktu. Hasilnya mengejutkan — **ada kekosongan total selama beberapa hari** (5–12 Desember 2023). Tidak ada satu pun transaksi.

Pertanyaannya: kenapa? Libur? Hujan deras? Atau data yang hilang?

```
Hipotesis 1: Hujan lebat → cek indikator hujan → tidak ada lonjakan
Hipotesis 2: Cuaca ekstrem → cek kelembapan & awan → tidak ada pola
Kesimpulan: Bukan fenomena nyata — ini data yang hilang (tidak tercatat)
```

Dengan investigasi ini, saya tidak serta-merta drop data atau membiarkannya merusak analisis. Gap ditangani dengan **interpolasi linear + rolling mean** agar time series tetap bisa dibaca. Tanpa langkah ini, insight tentang pola waktu bisa menyesatkan.

---

### Cuaca: kelihatannya penting, ternyata tidak sesederhana itu

Intuisi awal: cuaca buruk → biaya naik. Logis.

Tapi setelah saya plot dan hitung korelasi, **tidak ada hubungan linear antara variabel cuaca dengan biaya pengiriman** — baik suhu, kelembapan, kecepatan angin, maupun kondisi awan.

Apakah cuaca tidak berpengaruh sama sekali? Belum tentu.

Saya lanjut dengan clustering berbasis cuaca menggunakan **PCA + K-Means** (karena 5 variabel cuaca terlalu banyak untuk dianalisis satu-satu):

| Klaster Cuaca | Kondisi | Pola Biaya |
|---|---|---|
| 0 | Stabil, lembap, awan tebal | Moderat — pengiriman efisien |
| 1 | Kering, lebih dingin | Konsisten — tidak banyak variasi |
| 2 | Angin kencang, awan sedikit | Tidak berubah signifikan |
| 3 | Hujan, angin, suhu ekstrem | Bervariasi — ada yang tinggi, ada yang rendah |

**Kesimpulan:** Cuaca tidak langsung menentukan biaya, tapi ia membentuk *konteks* operasional. Kondisi cuaca stabil (Klaster 0 & 1) cenderung lebih efisien untuk pengiriman. Temuan ini tetap relevan secara bisnis meski tidak linear secara statistik.

> *Di sinilah saya memutuskan: variabel cuaca tidak di-drop, tapi digabung jadi satu komponen `combined_weather` via PCA — mengurangi noise tanpa kehilangan sinyal.*

---

### Aktivitas aplikasi: scatter plot yang membingungkan

Hipotesis wajar: semakin banyak orang pakai aplikasi → permintaan tinggi → biaya naik.

Tapi scatter plot-nya acak total. Aktivitas rendah bisa biaya tinggi, aktivitas tinggi bisa biaya rendah.

Ini bukan berarti variabel ini tidak penting — mungkin ada **segmen tersembunyi** di dalamnya. Saya pakai **Elbow Method** untuk tentukan k optimal (k=3), lalu K-Means:

| Klaster | Profil Pengguna | Interpretasi |
|---|---|---|
| 0 | Aktivitas rendah, biaya rendah | Pengguna yang price-sensitive, sering pakai promo |
| 1 | Aktivitas sedang, biaya tinggi | Pengguna loyal yang tinggal lebih jauh atau pesan di jam sibuk |
| 2 | Aktivitas tinggi, biaya sangat tinggi | Pengguna premium — high value, tidak sensitif harga |

Segmentasi ini jauh lebih informatif dari nilai mentahnya — dan langsung bisa diterjemahkan ke strategi bisnis.

---

### Kategori makanan & jarak: konfirmasi dan nuansa

Distribusi kategori makanan hampir merata, tapi ada **4 kategori dengan frekuensi sangat rendah** (9, 19, 29, 39). Kenapa?

Saya cek rata-rata jarak per kategori:

- Kategori frekuensi rendah → **jarak rata-rata lebih jauh**
- Kategori frekuensi tinggi → **jarak lebih dekat, biaya lebih terjangkau**

Artinya: orang tidak menghindari jenis makanan tertentu karena tidak suka — mereka menghindarinya karena **ongkos kirimnya mahal akibat jarak**. Ini insight yang bisa jadi dasar keputusan ekspansi restoran mitra.

---

### Kenaikan tarif: keberanian untuk drop fitur

Kolom `kenaikan_tarif_minimum` hampir seluruhnya bernilai sama. Secara statistik: **variabilitas nol, tidak ada informasi**.

Keputusan: drop. Model tidak akan belajar apa-apa dari fitur ini — memasukkannya justru menambah noise.

---

### Fitur waktu: diuji, tidak terbukti

Saya mencoba feature engineering dari `waktu_pemesanan`: jam, hari, bulan — dengan **cyclic encoding** (sin/cos) agar model mengenali pola siklus waktu. Dievaluasi dengan **TimeSeriesSplit** agar tidak terjadi data leakage.

Hasilnya: tidak ada peningkatan performa. Drop.

> *Fail fast adalah bagian dari proses yang sehat — lebih baik tahu lebih awal daripada mempertahankan fitur yang tidak berkontribusi.*

---

## 🤖 Modelling

Setelah insight terkumpul, saya bandingkan 5 model sekaligus dengan pendekatan yang konsisten:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | lebih tinggi | lebih tinggi | lebih rendah |
| Gradient Boosting | lebih tinggi | lebih tinggi | lebih rendah |
| XGBoost | lebih tinggi | lebih tinggi | lebih rendah |
| LightGBM | lebih tinggi | lebih tinggi | lebih rendah |
| **CatBoost ✅** | **792.40** | **573.06** | **0.992** |

CatBoost menang di semua metrik — mampu menjelaskan **99.2% variabilitas** data. Hyperparameter tuning dicoba tapi tidak signifikan; base model sudah cukup baik.

**Metrik: RMSE** — dipilih agar prediksi yang meleset jauh "dihukum" lebih berat, relevan untuk konteks pricing.

---

## 🏆 Apa yang Akhirnya Paling Berpengaruh?

Feature importance dari model mengkonfirmasi banyak temuan di atas:

```
1. kategori_makanan                              ████████████████████
2. cluster_aktivitas_aplikasi_biaya_pengiriman   ███████████████
3. cluster_combined_weather_biaya_pengiriman     ████████████
4. jarak_terdekat / jarak_rata2                  ██████████
```

Yang menarik: **fitur hasil clustering lebih informatif dari variabel mentahnya**. Ini validasi bahwa pendekatan segmentasi di EDA bukan hanya latihan akademis — ia memang menangkap sinyal yang tidak terlihat di raw data.

---

## 💡 Rekomendasi untuk Perusahaan

| Area | Rekomendasi |
|---|---|
| **Pelanggan** | Segmentasi berbasis klaster aktivitas — promo untuk Klaster 0, loyalty program untuk Klaster 1, exclusive membership untuk Klaster 2 |
| **Operasional** | Prioritaskan pengiriman saat cuaca stabil (Klaster 0 & 1); siapkan protokol khusus saat ekstrem |
| **Ekspansi** | Tambah mitra restoran di kategori makanan dengan jarak jauh — potensi demand ada, tapi terhalang ongkir |
| **Logistik** | Kelompokkan kategori makanan berdasarkan kebutuhan penanganan, bukan hanya jenis kuliner |

---

## 🛠️ Tech Stack

```python
pandas · numpy · scikit-learn · seaborn · matplotlib · scipy
xgboost · lightgbm · catboost · joblib
```

---

## ▶️ Cara Menjalankan

```bash
pip install catboost lightgbm xgboost scikit-learn pandas numpy seaborn matplotlib scipy
```

Jalankan `POJOKSTATISTIK_Binar.ipynb` dari atas ke bawah. Dataset otomatis diunduh saat cell *Loading Data* dieksekusi.
