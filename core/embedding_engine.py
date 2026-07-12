import numpy as np
import pandas as pd
import streamlit as st

from core.gemini_client import get_gemini_client
from utils.data_loader import load_catalog, get_catalog_embeddings

# 5 famili aroma berdasarkan Fragrance Wheel (Michael Edwards).

FAMILI_AROMA = {
    "fresh": "aroma segar seperti jeruk, daun hijau, dan kesegaran air",
    "floral": "aroma bunga-bungaan yang lembut dan feminin",
    "fruity": "aroma buah-buahan yang manis dan ceria",
    "woody": "aroma kayu-kayuan yang hangat dan maskulin",
    "oriental": "aroma rempah, vanila, atau kopi yang hangat dan mewah",
}


def _susun_kalimat_preferensi(skor: dict[str, int]) -> str:
    """Ubah skor slider (1-5 per famili) jadi 1 kalimat naratif.

    Kenapa diubah jadi kalimat (bukan dikirim sebagai angka mentah ke
    embedding model)? Karena model embedding dilatih untuk memahami
    BAHASA, bukan angka. "Fresh: 5, Woody: 4" tidak akan diproses
    semaknanya seperti kalimat naratif yang dipahami secara kontekstual.

    Hanya famili dengan skor >= 3 yang disebutkan secara eksplisit
    sebagai preferensi kuat -- ini mencegah kalimat jadi terlalu panjang
    dan "encer" maknanya kalau semua famili disebut rata.
    """
    bagian_kuat = [
        FAMILI_AROMA[famili]
        for famili, nilai in skor.items()
        if nilai >= 3
    ]

    if not bagian_kuat:
        return "aroma yang seimbang dan tidak terlalu menonjol di satu sisi"

    return "Saya suka parfum dengan " + ", dan juga ".join(bagian_kuat) + "."


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Hitung cosine similarity antara 2 vector.

    Cosine similarity mengukur sudut antar vector (bukan jaraknya),
    sehingga tidak terpengaruh oleh "panjang" vector -- cocok untuk
    membandingkan makna semantik, dimana yang penting adalah ARAH
    vector di ruang embedding, bukan magnitude-nya.

    Hasil: -1 (berlawanan total) sampai 1 (identik). Untuk teks sehari-hari
    biasanya hasilnya di rentang 0.3 - 0.9.

    PUBLIK (tanpa underscore) karena dipakai ulang lintas modul --
    Tab 2 (embedding_engine) dan Tab 3 (vision_engine, untuk pencocokan
    hasil deskripsi Gemini Vision ke katalog) sama-sama butuh fungsi ini,
    supaya logikanya konsisten dan tidak ada kode duplikat."""
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _filter_harga(df: pd.DataFrame, harga_min: int | None, harga_max: int | None) -> pd.DataFrame:
    """Filter katalog berdasarkan range harga user (hard constraint).

    Sama seperti di recommender_engine.py: budget adalah aturan PASTI,
    bukan sesuatu yang perlu "dipertimbangkan" secara semantik. Difilter
    SEBELUM penghitungan similarity, supaya produk di luar budget tidak
    ikut dibandingkan sama sekali (lebih murni & lebih cepat daripada
    menghitung similarity semua produk lalu baru disaring belakangan).
    """
    if harga_min is not None:
        df = df[df["harga"] >= harga_min]
    if harga_max is not None:
        df = df[df["harga"] <= harga_max]
    return df


def cari_rekomendasi_berdasarkan_notes(
    skor_preferensi: dict[str, int],
    top_n: int = 5,
    harga_min: int | None = None,
    harga_max: int | None = None,
) -> pd.DataFrame:
    """Fungsi utama yang dipanggil dari Tab 2 di app.py.

    Args:
        skor_preferensi: dict seperti {"fresh": 5, "floral": 2, "fruity": 1,
                          "woody": 4, "oriental": 2} -- dari slider user.
        top_n: jumlah rekomendasi teratas yang dikembalikan.
        harga_min, harga_max: batas budget user dalam Rupiah (angka).
                               None = tidak ada batas di sisi itu.

    Returns:
        DataFrame berisi top_n produk yang paling mirip, sudah diurutkan,
        dengan kolom tambahan "similarity_score" untuk transparansi
        (supaya user/dosen bisa lihat seberapa "yakin" sistemnya).
    """
    client = get_gemini_client()
    df = load_catalog()
    catalog_embeddings = get_catalog_embeddings()

    if not catalog_embeddings:
       
        return pd.DataFrame()

    # Filter budget DULU, sebelum menghitung similarity -- produk di luar
    df_filtered = _filter_harga(df, harga_min, harga_max)
    if df_filtered.empty:
        st.warning(
            "Tidak ada produk dalam range harga ini. Coba lebarkan budget kamu."
        )
        return pd.DataFrame()

    kalimat_user = _susun_kalimat_preferensi(skor_preferensi)
    vector_user = client.embed_text(kalimat_user)

    if not vector_user:
        st.warning("Gagal memproses preferensi kamu, coba lagi.")
        return pd.DataFrame()

    # Hitung similarity user vs SETIAP produk yang lolos filter harga
    hasil = []
    for _, row in df_filtered.iterrows():
        id_produk = row["id_produk"]
        vector_produk = catalog_embeddings.get(id_produk)
        if vector_produk is None:
            continue  # produk ini gagal di-embed sebelumnya, skip saja
        skor = cosine_similarity(vector_user, vector_produk)
        hasil.append({"id_produk": id_produk, "similarity_score": skor})

    df_skor = pd.DataFrame(hasil)
    if df_skor.empty:
        return pd.DataFrame()

    # Gabungkan skor similarity ke data produk lengkap, urutkan, ambil top_n
    df_gabung = df_filtered.merge(df_skor, on="id_produk")
    df_gabung = df_gabung.sort_values("similarity_score", ascending=False)

    return df_gabung.head(top_n).reset_index(drop=True)