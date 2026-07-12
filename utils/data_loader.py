"""
utils/data_loader.py
---------------------
Bertugas memuat data_parfum.csv dan menyiapkan embedding untuk Tab 2.

KONSEP PENTING -- kenapa ada 2 fungsi cache yang berbeda:
1. load_catalog()      -> baca CSV (cepat, cuma baca file)
2. get_catalog_embeddings() -> panggil Gemini API untuk SELURUH produk
   (LAMBAT dan PAKAI KUOTA API), makanya WAJIB di-cache supaya hanya
   dihitung SEKALI selama app berjalan, bukan setiap user berinteraksi.

@st.cache_data dipakai (bukan @st.cache_resource) karena hasil yang
disimpan berupa DATA (DataFrame, list angka), bukan OBJECT/KONEKSI.
Aturan praktis: cache_resource untuk koneksi/client, cache_data untuk hasil.

CATATAN UNTUK DATASET BARU (492 produk, sebelumnya 15):
Gemini API (BatchEmbedContentsRequest) punya HARD LIMIT maksimal 100 item
per 1 panggilan batch -- ini bukan soal performa/kuota, tapi validasi
keras dari Google sendiri (kalau dilanggar, API langsung menolak dengan
error 400 INVALID_ARGUMENT, bukan sekadar lambat).

Karena katalog sekarang 492 produk (lebih dari 100), get_catalog_embeddings()
TIDAK BISA lagi mengirim semua sekaligus dalam 1 batch seperti versi
sebelumnya. Solusinya: katalog dipecah jadi beberapa CHUNK (potongan)
berukuran maksimal 100 baris, masing-masing chunk dikirim sebagai batch
terpisah, lalu hasilnya digabung. Tetap jauh lebih hemat dibanding embed
satu-satu per baris (492 request individual) -- sekarang jadi sekitar
5 request batch saja (492 / 100, dibulatkan ke atas).

CATATAN TAMBAHAN -- soal HEMAT KUOTA saat rate limit (429) tersisa:
Versi sebelumnya, kalau SATU chunk gagal di tengah jalan (misal chunk ke-3
dari 5 kena rate limit), seluruh hasil (termasuk chunk 1 & 2 yang sudah
BERHASIL didapat) dibuang percuma lewat `return {}`. Karena hasilnya kosong,
Streamlit TIDAK menyimpannya sebagai cache valid, sehingga percobaan
berikutnya mengulang chunk 1 dari awal -- buang-buang kuota yang sudah
"terpakai" untuk chunk yang sebenarnya sudah sukses.

Sekarang: chunk yang BERHASIL tetap disimpan dan dikembalikan apa adanya,
chunk yang GAGAL ditandai lewat pesan error yang jelas (bukan exception
yang membuat seluruh fungsi gagal), supaya pengguna tahu hasilnya belum
lengkap, tapi tidak ada kuota yang terbuang sia-sia untuk chunk yang
sebenarnya sudah berhasil.
"""

import streamlit as st
import pandas as pd

from core.gemini_client import get_gemini_client

CSV_PATH = "data/data_parfum.csv"

MAKS_ITEM_PER_BATCH_EMBEDDING = 100


@st.cache_data
def load_catalog() -> pd.DataFrame:
    """Baca data_parfum.csv. Di-cache karena file tidak berubah selama
    app berjalan, jadi tidak perlu dibaca ulang dari disk tiap rerun."""
    df = pd.read_csv(CSV_PATH, sep=";")

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    df["waktu"] = df["waktu"].astype(str).str.strip().str.lower()
    return df


def format_harga(angka) -> str:
    """Ubah angka harga (249000) jadi teks rapi 'Rp 249.000'.

    Dipakai di kartu produk (ui_components.py) supaya tampilan harga
    konsisten dan mudah dibaca, terlepas dari format aslinya di CSV
    (CSV baru menyimpan harga sebagai teks 'Rp249.000,00').
    """
    if pd.isna(angka):
        return "Harga tidak tersedia"
    # f"{angka:,.0f}" -> "249,000" (pemisah koma ala Inggris), lalu kita
    # ganti koma jadi titik supaya sesuai format ribuan Indonesia.
    return f"Rp {angka:,.0f}".replace(",", ".")


def _teks_untuk_embedding(row: pd.Series) -> str:
    """Gabungkan kolom-kolom relevan jadi 1 teks yang dikirim ke embedding model.

    Kenapa gabung notes + deskripsi_wangi (bukan salah satu saja)?
    - deskripsi_wangi: menangkap KESAN/feel aroma secara naratif (di CSV
      baru ini berisi narasi awal-tengah-akhir semprotan)
    - top/middle/base notes: menangkap KOMPOSISI teknis aroma
    Gabungan keduanya membuat embedding lebih kaya makna dibanding salah satunya saja.
    """
    return (
        f"{row['deskripsi_wangi']} "
        f"Top notes: {row['top_notes']}. "
        f"Middle notes: {row['middle_notes']}. "
        f"Base notes: {row['base_notes']}."
    )


def _pecah_jadi_chunk(daftar: list, ukuran_chunk: int) -> list[list]:
    """Pecah satu list panjang jadi beberapa list lebih kecil (chunk),
    masing-masing maksimal berisi ukuran_chunk item.

    Contoh: _pecah_jadi_chunk([1,2,3,4,5], 2) -> [[1,2], [3,4], [5]]
    (chunk terakhir boleh lebih kecil kalau jumlahnya tidak pas habis dibagi).

    Dipakai untuk memenuhi limit 100 item/batch dari Gemini API -- generik
    (tidak terikat tipe data tertentu) supaya bisa dipakai ulang kapan pun
    dibutuhkan logika "potong jadi beberapa bagian" yang serupa.
    """
    return [
        daftar[i : i + ukuran_chunk]
        for i in range(0, len(daftar), ukuran_chunk)
    ]


@st.cache_data(show_spinner="Menyiapkan embedding katalog parfum...")
def get_catalog_embeddings() -> dict[int, list[float]]:
    """Precompute embedding untuk SELURUH katalog, SEKALI saja.

    Return: dict {id_produk: vector_embedding}
    Pakai dict (bukan list/kolom DataFrame) supaya lookup by id_produk O(1)
    dan tidak tergantung urutan baris di CSV.

    PENTING: fungsi ini di-cache oleh Streamlit berdasarkan hash dari
    kode fungsi + argumennya. Karena tidak ada argumen yang berubah,
    fungsi ini hanya akan benar-benar berjalan SATU KALI per sesi server
    (bahkan across banyak user, kalau di-deploy), lalu hasilnya dipakai
    ulang dari cache. Ini yang membuat precompute jadi murah.

    DIPECAH JADI CHUNK (lihat MAKS_ITEM_PER_BATCH_EMBEDDING di atas file
    ini): katalog 492 produk tidak bisa dikirim sekaligus dalam 1 batch
    karena Gemini API menolak permintaan dengan lebih dari 100 item
    (error 400 INVALID_ARGUMENT). Jadi df dipecah jadi beberapa potongan
    kecil dulu, masing-masing di-embed lewat panggilan batch terpisah,
    baru semua hasilnya digabung jadi 1 dict di akhir.

    PENTING -- HEMAT KUOTA SAAT SEBAGIAN CHUNK GAGAL:
    Kalau ada chunk yang gagal (misal kena rate limit 429 setelah retry
    di gemini_client.py juga habis), fungsi ini TIDAK membuang chunk lain
    yang sudah berhasil. Dict yang dikembalikan berisi SEMUA produk yang
    berhasil di-embed sejauh itu, walau jumlahnya belum 100% lengkap.
    Karena hasilnya tidak kosong (asalkan minimal 1 chunk berhasil),
    Streamlit AKAN menyimpannya di cache -- jadi percobaan berikutnya
    tidak perlu mengulang chunk yang sudah sukses dari awal lagi.

    Pesan st.warning (bukan st.error+return{}) dipakai untuk chunk yang
    gagal, supaya user tetap tahu ada bagian katalog yang belum lengkap,
    tapi fungsi tetap mengembalikan hasil sebanyak yang berhasil didapat.
    """
    df = load_catalog()
    client = get_gemini_client()

    teks_list = [_teks_untuk_embedding(row) for _, row in df.iterrows()]
    id_list = df["id_produk"].tolist()

    chunk_teks = _pecah_jadi_chunk(teks_list, MAKS_ITEM_PER_BATCH_EMBEDDING)
    chunk_id = _pecah_jadi_chunk(id_list, MAKS_ITEM_PER_BATCH_EMBEDDING)

    semua_vector: dict[int, list[float]] = {}
    jumlah_chunk_gagal = 0

    for nomor_chunk, (teks_chunk, id_chunk) in enumerate(zip(chunk_teks, chunk_id), start=1):
        vectors_chunk = client.embed_batch(teks_chunk)

        if not vectors_chunk or len(vectors_chunk) != len(teks_chunk):
            jumlah_chunk_gagal += 1
            continue

        semua_vector.update(zip(id_chunk, vectors_chunk))

    if jumlah_chunk_gagal > 0:
        total_chunk = len(chunk_teks)
        st.warning(
            f"{jumlah_chunk_gagal} dari {total_chunk} bagian katalog gagal "
            f"di-embed (kemungkinan kuota API sedang penuh). "
            f"{len(semua_vector)} dari {len(df)} produk tetap berhasil "
            f"diproses dan bisa langsung dipakai. Refresh halaman beberapa "
            f"saat lagi untuk melengkapi sisanya."
        )

    if not semua_vector:
        st.error(
            "Gagal precompute seluruh embedding katalog. Cek koneksi "
            "Gemini API atau coba refresh halaman beberapa saat lagi."
        )

    return semua_vector