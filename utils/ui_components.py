import streamlit as st
import pandas as pd

from utils.data_loader import format_harga


def _tampilkan_foto_produk(foto_url: str | None):
    """Tampilkan foto produk dari URL. Kalau URL kosong/invalid, tampilkan
    kotak placeholder bergaris putus-putus -- BUKAN error -- karena di
    dataset ini banyak produk yang fotonya memang belum diisi (akan
    dilengkapi user setelah web jadi)."""

    if foto_url and isinstance(foto_url, str) and foto_url.startswith("http"):
        try:
            st.image(foto_url, use_container_width=True)
            return
        except Exception:
            pass

    st.markdown(
        "<div style='background:#3332;border-radius:8px;"
        "padding:40px 0;text-align:center;font-size:12px;"
        "color:#666;'>Foto belum<br>tersedia</div>",
        unsafe_allow_html=True,
    )


def tampilkan_kartu_produk(row: pd.Series, skor_label: str | None = None):
    """Tampilkan 1 kartu produk lengkap: foto, nama, brand, notes, harga,
    deskripsi, dan tombol beli yang mengarah ke link toko asli.

    Args:
        row: 1 baris dari DataFrame katalog (harus punya semua kolom CSV).
        skor_label: teks skor yang ditampilkan di pojok kanan, misal "81%"
                    dari similarity_score (Tab 2) atau "Rekomendasi #1" (Tab 1).
                    Kalau None, bagian skor tidak ditampilkan.
    """
    with st.container(border=True):
        col_foto, col_info, col_skor = st.columns([1, 3, 1])

        with col_foto:
            _tampilkan_foto_produk(row.get("foto_url"))

        with col_info:
            st.markdown(f"**{row['nama_parfum']}** oleh {row['brand']}")
            st.caption(
                f"{row['kategori']} &middot; {format_harga(row['harga'])} "
                f"&middot; {row['gender']}"
            )

            st.write(row["deskripsi_wangi"])

            st.markdown(
                f"<small>"
                f"<b>Top:</b> {row['top_notes']} &nbsp;|&nbsp; "
                f"<b>Middle:</b> {row['middle_notes']} &nbsp;|&nbsp; "
                f"<b>Base:</b> {row['base_notes']}"
                f"</small>",
                unsafe_allow_html=True,
            )

            link_toko = row.get("link_toko")
            if isinstance(link_toko, str) and link_toko.startswith("http"):
                st.link_button("Beli di toko", link_toko, type="secondary")

        with col_skor:
            if skor_label:
                st.metric("Kecocokan", skor_label)


def tampilkan_daftar_produk(df: pd.DataFrame, kolom_skor: str | None = None):
    """Tampilkan banyak kartu produk sekaligus dari sebuah DataFrame hasil rekomendasi.

    Args:
        df: DataFrame hasil dari embedding_engine atau recommender_engine.
        kolom_skor: nama kolom yang berisi skor (misal "similarity_score").
                    Kalau diisi, skor itu diformat jadi persen dan ditampilkan.
                    Kalau None, kartu ditampilkan tanpa skor.
    """
    for _, row in df.iterrows():
        skor_label = None
        if kolom_skor and kolom_skor in row:
            skor_label = f"{row[kolom_skor]:.0%}"
        tampilkan_kartu_produk(row, skor_label=skor_label)