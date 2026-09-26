import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman
st.set_page_config(
    page_title="E-Commerce Customer RFM Segmentation",
    page_icon="🛒",
    layout="wide"
)

# Header Aplikasi
st.title("🛒 Dashboard Segmentasi Pelanggan E-Commerce")
st.markdown("""
Aplikasi analitik prediktif berbasis **RFM (Recency, Frequency, & Monetary)** menggunakan algoritma **K-Means Clustering**.
Sistem ini memprediksi segmen pelanggan baru dan memberikan rekomendasi strategi pemasarannya secara otomatis.
""")

# Load Model & Scaler
@st.cache_resource
def load_rfm_artifacts():
    model = joblib.load('model_kmeans_rfm.pkl')
    scaler = joblib.load('scaler_rfm.pkl')
    return model, scaler

try:
    model, scaler = load_rfm_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file model_kmeans_rfm.pkl dan scaler_rfm.pkl berada di folder yang sama. Error: {e}")
    st.stop()

# Mapping Informasi Bisnis Sesuai Profil Cluster
info_segmen = {
    1: {
        'nama': 'Champions / Sultan (Sangat Aktif & Belanja Besar)',
        'strategi': 'Berikan perlakuan VIP, akses eksklusif ke produk baru pertama kali, dan reward program loyalitas tanpa perlu diskon harga.'
    },
    2: {
        'nama': 'Loyal Customers (Pelanggan Rutin & Potensial)',
        'strategi': 'Tawarkan bundling produk, diskon bersyarat (misal: belanja min. $1000), dan rekomendasi produk terkait (upselling).'
    },
    0: {
        'nama': 'Promising / New Customers (Pelanggan Baru/Tumbuh)',
        'strategi': 'Kirimkan voucher diskon untuk order kedua dan newsletter katalog produk terlaris agar terdorong belanja ulang.'
    },
    3: {
        'nama': 'Lost / Hibernating (Pasif & Sudah Lama Tidak Belanja)',
        'strategi': 'Kirimkan kampanye email win-back "We Miss You" disertai kupon diskon berbatas waktu (FOMO) untuk reaktivasi akun.'
    }
}

# Form Input Metrik RFM
st.subheader("📝 Masukkan Parameter Transaksi Pelanggan Baru:")
col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input(
        "Recency (Hari sejak belanja terakhir):",
        min_value=1,
        max_value=1000,
        value=15,
        help="Semakin kecil angkanya, berarti pelanggan baru saja aktif berbelanja."
    )

with col2:
    frequency = st.number_input(
        "Frequency (Jumlah kali transaksi order):",
        min_value=1,
        max_value=500,
        value=10,
        help="Berapa kali pelanggan membuat pesanan belanja."
    )

with col3:
    monetary = st.number_input(
        "Monetary (Total uang belanja dalam $):",
        min_value=1.0,
        max_value=500000.0,
        value=5000.0,
        step=50.0,
        help="Total akumulasi uang yang dibelanjakan pelanggan."
    )

# Tombol Prediksi
if st.button("🚀 Analisis Segmen Pelanggan", type="primary", use_container_width=True):
    # 1. Preprocessing (Log transform + Scaling)
    input_log = np.log1p([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_log)
    
    # 2. Prediksi Cluster
    cluster_pred = int(model.predict(input_scaled)[0])
    segmen_detail = info_segmen.get(cluster_pred, {
        'nama': f'Cluster {cluster_pred}',
        'strategi': 'Lakukan analisis perilaku belanja secara berkala.'
    })
    
    st.markdown("---")
    res1, res2 = st.columns([1, 2])
    with res1:
        st.metric(label="Cluster Model", value=f"Cluster {cluster_pred}")
        st.subheader(segmen_detail['nama'])
    with res2:
        st.info(f"💡 **Rekomendasi Tindakan Pemasaran:**\n\n{segmen_detail['strategi']}")
 
# Tampilan Data Hasil Segmentasi
st.markdown("---")
st.subheader("📂 Cuplikan Database Hasil Segmentasi Pelanggan")
try:
    df_hasil = pd.read_csv('hasil_segmentasi_pelanggan_rfm.csv')
    st.dataframe(df_hasil.head(10), use_container_width=True)
    st.caption(f"Menampilkan 10 dari total {len(df_hasil)} pelanggan yang berhasil disegmentasi.")
except Exception:
    st.info("File 'hasil_segmentasi_pelanggan_rfm.csv' belum ditemukan. Jalankan sel ekspor di notebook untuk menampilkannya di sini.")