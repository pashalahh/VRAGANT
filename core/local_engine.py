"""
core/local_engine.py
---------------------
Mesin rekomendasi LOKAL (offline) untuk ScentDaily dan ScentNote.
Tidak butuh internet atau API key -- semua perhitungan dilakukan
di lokal menggunakan model TF-IDF yang sudah dilatih dari dataset.

PERBEDAAN DENGAN VERSI GEMINI:
- Versi Gemini (recommender_engine.py / embedding_engine.py):
  mengirim data ke server Google, Gemini "bernalar" atau menghitung
  embedding menggunakan model raksasa yang dilatih triliunan data.
- Versi Lokal (file ini):
  semua perhitungan di komputer sendiri. TF-IDF mengukur FREKUENSI
  kata penting dari setiap deskripsi produk, lalu cosine similarity
  menghitung jarak matematis antar vektor -- tanpa cloud, tanpa biaya.

CARA KERJA TF-IDF SINGKAT:
TF (Term Frequency) = seberapa sering kata muncul di 1 dokumen.
IDF (Inverse Document Frequency) = seberapa JARANG kata muncul di
seluruh koleksi dokumen (kata langka = lebih informatif).
TF-IDF = TF × IDF → angka tinggi berarti kata itu PENTING dan UNIK
untuk dokumen itu. Hasilnya tiap produk jadi sebuah vektor angka,
lalu cosine similarity mengukur "sudut" antar vektor (semakin kecil
sudutnya, semakin mirip maknanya).
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_catalog

MODEL_DIR = "models"
VECTORIZER_PATH = f"{MODEL_DIR}/tfidf_vectorizer.pkl"
MATRIX_PATH = f"{MODEL_DIR}/tfidf_matrix.npy"
IDS_PATH = f"{MODEL_DIR}/produk_ids.npy"


@st.cache_resource
def _load_model():
    """Load model TF-IDF dari disk, sekali saja per sesi.

    @st.cache_resource (bukan @st.cache_data) karena kita menyimpan
    OBJECT Python (vectorizer, array besar) -- bukan data serializable
    biasa. cache_resource memastikan object ini dibuat sekali dan
    di-share ke semua user tanpa di-copy ulang tiap rerun.
    """
    try:
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        matrix = np.load(MATRIX_PATH)
        produk_ids = np.load(IDS_PATH)
        return vectorizer, matrix, produk_ids
    except FileNotFoundError:
        st.error(
            "Model lokal tidak ditemukan. Pastikan folder 'models/' berisi "
            "tfidf_vectorizer.pkl, tfidf_matrix.npy, dan produk_ids.npy. "
            "Jalankan notebook Colab atau script training dulu."
        )
        return None, None, None


def _cosine_similarity_manual(vec_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Hitung cosine similarity antara 1 vektor query dengan semua baris matrix.

    Lebih efisien dari menghitung satu-satu karena numpy melakukan
    operasi ini secara paralel (vectorized).

    vec_a shape: (n_fitur,)
    mat_b shape: (n_produk, n_fitur)
    return shape: (n_produk,) -- satu skor per produk
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(mat_b, axis=1)
    if norm_a == 0:
        return np.zeros(len(mat_b))
    # dot product tiap baris mat_b dengan vec_a, dibagi norm keduanya
    return mat_b.dot(vec_a) / (norm_b * norm_a + 1e-10)


def _filter_harga(df: pd.DataFrame, harga_min, harga_max) -> pd.DataFrame:
    """Filter dataframe berdasarkan range harga (hard constraint)."""
    if harga_min:
        df = df[df["harga"] >= harga_min]
    if harga_max:
        df = df[df["harga"] <= harga_max]
    return df


# ===========================================================================
# SCENTDAILY LOKAL
# Cara kerja:
# 1. Filter katalog berdasarkan gender + harga (hard constraint)
# 2. Susun "kalimat query" dari pilihan user (aktivitas + waktu + lokasi
#    + deskripsi custom kalau ada)
# 3. Transform kalimat query jadi vektor TF-IDF
# 4. Hitung cosine similarity query vs setiap produk yang lolos filter
# 5. Ambil top_n produk dengan skor tertinggi
# ===========================================================================

def cari_lokal_berdasarkan_aktivitas(
    aktivitas: list[str],
    waktu: list[str],
    gender: str,
    lokasi: str,
    deskripsi_custom: str = "",
    harga_min=None,
    harga_max=None,
    jumlah_rekomendasi: int = 5,
) -> pd.DataFrame:
    """Cari rekomendasi parfum berbasis aktivitas menggunakan TF-IDF lokal.

    Tidak butuh internet/API. Hasilnya tidak sedetail Gemini (tidak ada
    alasan personal per produk), tapi konsisten dan bisa jalan offline.

    Returns:
        DataFrame produk rekomendasi dengan kolom tambahan 'skor_lokal'
        (cosine similarity, 0.0-1.0). DataFrame kosong kalau gagal.
    """
    vectorizer, matrix, produk_ids = _load_model()
    if vectorizer is None:
        return pd.DataFrame()

    df = load_catalog()

    # Filter gender (hard constraint)
    df_filtered = df[df["gender"] == gender]

    df_filtered = _filter_harga(df_filtered, harga_min, harga_max)
    if df_filtered.empty:
        st.warning("Tidak ada produk yang cocok dengan filter gender/harga.")
        return pd.DataFrame()

    # Susun kalimat query dari pilihan user
    bagian_query = []
    bagian_query.extend([a.lower() for a in aktivitas])
    bagian_query.extend([w.lower() for w in waktu])
    bagian_query.append(lokasi.lower())
    if deskripsi_custom and deskripsi_custom.strip():
        bagian_query.append(deskripsi_custom.strip())
    query = " ".join(bagian_query)

    # Transform query ke vektor TF-IDF menggunakan vocabulary yang sama
    vec_query = vectorizer.transform([query]).toarray()[0]

    # Hitung similarity hanya untuk produk yang lolos filter
    id_lolos = df_filtered["id_produk"].values
    idx_lolos = np.where(np.isin(produk_ids, id_lolos))[0]
    matrix_lolos = matrix[idx_lolos]

    skor = _cosine_similarity_manual(vec_query, matrix_lolos)

    df_skor = pd.DataFrame({"id_produk": produk_ids[idx_lolos], "skor_lokal": skor})
    df_hasil = df_filtered.merge(df_skor, on="id_produk")
    df_hasil = df_hasil.sort_values("skor_lokal", ascending=False)

    print(f"\n[LOKAL LOG] Query: '{query}'")
    print("[LOKAL LOG] Top 5 skor cosine similarity (ScentDaily lokal):")
    for _, row in df_hasil.head(5).iterrows():
        print(f"  {row['skor_lokal']:.4f}  |  {row['nama_parfum']} ({row['brand']})")
    print()

    return df_hasil.head(jumlah_rekomendasi).reset_index(drop=True)


# ===========================================================================
# SCENTNOTE LOKAL
# Cara kerja:
# 1. Terjemahkan skor slider famili aroma jadi kalimat query
#    (sama persis dengan embedding_engine.py, supaya bisa dibandingkan)
# 2. Transform kalimat ke vektor TF-IDF
# 3. Filter harga (hard constraint)
# 4. Hitung cosine similarity ke seluruh katalog yang lolos filter
# 5. Ambil top_n
# ===========================================================================

FAMILI_AROMA = {
    "fresh": "aroma segar seperti jeruk daun hijau kesegaran air citrus aquatic",
    "floral": "aroma bunga-bungaan rose jasmine lembut feminin floral",
    "fruity": "aroma buah-buahan manis ceria berry peach apple fruity",
    "woody": "aroma kayu-kayuan hangat maskulin cedar sandalwood oud woody",
    "oriental": "aroma rempah vanila kopi hangat mewah amber musk gourmand oriental",
}


def _susun_kalimat_preferensi(skor: dict[str, int]) -> str:
    """Sama seperti di embedding_engine.py -- ubah skor slider jadi kalimat.
    Diduplikat di sini (bukan diimpor) supaya local_engine tidak bergantung
    pada file embedding_engine yang butuh koneksi Gemini untuk berjalan."""
    bagian_kuat = [
        FAMILI_AROMA[famili]
        for famili, nilai in skor.items()
        if nilai >= 3
    ]
    if not bagian_kuat:
        return "aroma yang seimbang dan tidak terlalu menonjol di satu sisi"
    return "Saya suka parfum dengan " + " dan juga ".join(bagian_kuat)


def cari_lokal_berdasarkan_notes(
    skor_preferensi: dict[str, int],
    top_n: int = 5,
    harga_min=None,
    harga_max=None,
) -> pd.DataFrame:
    """Cari rekomendasi parfum berbasis preferensi aroma menggunakan TF-IDF lokal.

    Konsep sama dengan embedding_engine.py (Gemini), bedanya:
    - Gemini embedding: model neural 768 dimensi, paham MAKNA kata secara
      kontekstual (tahu "segar" = "fresh" = "citrus" dst tanpa dikasih tahu)
    - TF-IDF lokal: model statistik 3000 dimensi, cocokkan KATA LITERAL
      (hanya cocok kalau kata yang sama muncul di query DAN di deskripsi)
    Makanya query TF-IDF di atas sengaja diisi sinonim secara eksplisit
    ("segar jeruk citrus aquatic") supaya tetap bisa mencocokkan walaupun
    kata yang dipakai berbeda.

    Returns:
        DataFrame produk dengan kolom tambahan 'similarity_score'
        (nama kolom sengaja sama dengan versi Gemini supaya ui_components
        bisa menampilkannya tanpa modifikasi -- interface tetap konsisten).
    """
    vectorizer, matrix, produk_ids = _load_model()
    if vectorizer is None:
        return pd.DataFrame()

    df = load_catalog()
    df_filtered = _filter_harga(df, harga_min, harga_max)
    if df_filtered.empty:
        st.warning("Tidak ada produk dalam range harga ini.")
        return pd.DataFrame()

    kalimat = _susun_kalimat_preferensi(skor_preferensi)
    vec_query = vectorizer.transform([kalimat]).toarray()[0]

    id_lolos = df_filtered["id_produk"].values
    idx_lolos = np.where(np.isin(produk_ids, id_lolos))[0]
    matrix_lolos = matrix[idx_lolos]

    skor = _cosine_similarity_manual(vec_query, matrix_lolos)

    df_skor = pd.DataFrame({"id_produk": produk_ids[idx_lolos], "similarity_score": skor})
    df_hasil = df_filtered.merge(df_skor, on="id_produk")
    df_hasil = df_hasil.sort_values("similarity_score", ascending=False)

    print(f"\n[LOKAL LOG] Kalimat preferensi: '{kalimat}'")
    print("[LOKAL LOG] Top 5 skor (ScentNote lokal):")
    for _, row in df_hasil.head(5).iterrows():
        print(f"  {row['similarity_score']:.4f}  |  {row['nama_parfum']} ({row['brand']})")
    print()

    return df_hasil.head(top_n).reset_index(drop=True)