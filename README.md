# 🛒 E-Commerce Customer Segmentation (RFM + K-Means)

Proyek Data Science end-to-end berbasis metodologi **CRISP-DM** untuk mengelompokkan pelanggan e-commerce menggunakan metrik **RFM (Recency, Frequency, Monetary)** dan algoritma **K-Means Clustering**.

---

## 📌 Gambaran Proyek
- **Dataset:** Log transaksi e-commerce (4.338 pelanggan unik).
- **Preprocessing:** Pembersihan transaksi retur/batal, penanganan nilai pencilan ekstrem (*skewness*) dengan Log Transformation (`np.log1p`), serta standarisasi fitur (`StandardScaler`).
- **Modeling:** K-Means Clustering ($k=4$) dievaluasi dengan *Elbow Method* dan *Silhouette Analysis*.
- **Deployment:** Antarmuka kalkulator prediksi segmen interaktif berbasis **Streamlit**.

---

## 👥 Hasil Profil Segmen & Rekomendasi Bisnis
1. **Champions / Sultan (Cluster 1):** Pelanggan belanja sangat sering dan bernilai tinggi $\rightarrow$ Reward VIP & akses awal produk baru tanpa diskon harga.
2. **Loyal Customers (Cluster 2):** Pelanggan rutin dengan transaksi berkala $\rightarrow$ Program poin & penawaran paket bundling (*upselling*).
3. **Promising / New Customers (Cluster 0):** Pelanggan baru/tumbuh $\rightarrow$ Voucher diskon pesanan kedua & katalog produk terlaris.
4. **Lost / Hibernating (Cluster 3):** Pelanggan pasif yang lama tidak belanja $\rightarrow$ Kampanye retensi *win-back* email "We Miss You" dengan kupon berbatas waktu.

---

## 🚀 Cara Menjalankan Aplikasi Secara Lokal
```bash
# 1. Clone repositori ini
git clone [https://github.com/USERNAME-KAMU/ecommerce-customer-segmentation-rfm.git](https://github.com/USERNAME-KAMU/ecommerce-customer-segmentation-rfm.git)

# 2. Masuk ke direktori proyek
cd ecommerce-customer-segmentation-rfm

# 3. Install seluruh pustaka dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi Streamlit
streamlit run app.py
