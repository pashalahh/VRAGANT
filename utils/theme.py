import base64
import datetime

import streamlit as st

WARNA_OREN_UTAMA = "#993C1D"      # judul italic "Virtual Fragrance Consultant", border placeholder logo
WARNA_OREN_MUDA = "#F0997B"       # border dashed placeholder logo
WARNA_OREN_HOVER = "#7A2F16"      # versi lebih gelap dari WARNA_OREN_UTAMA, untuk efek hover

WARNA_HIJAU_BORDER = "#BFE3C9"
WARNA_HIJAU_BG = "#F5FBF6"

CSS_TEMA_CERAH = f"""
<style>
/* Tombol "secondary" (dipakai utamanya oleh st.link_button "Beli di toko")
   diberi warna golden hour -- supaya jelas terlihat dan konsisten dengan
   tema, bukan abu-abu polos bawaan Streamlit. */
button[kind="secondary"], a[kind="secondary"] {{
    background-color: {WARNA_OREN_UTAMA} !important;
    color: white !important;
    border: 1px solid {WARNA_OREN_UTAMA} !important;
}}
button[kind="secondary"]:hover, a[kind="secondary"]:hover {{
    background-color: {WARNA_OREN_HOVER} !important;
    border: 1px solid {WARNA_OREN_HOVER} !important;
    color: white !important;
}}

/* Tombol "primary" (misal "Cari rekomendasi aktivitas") tetap dikuatkan
   warnanya juga supaya satu keluarga warna dengan tombol secondary,
   tapi sedikit lebih muda agar tetap terlihat beda level penekanannya. */
button[kind="primary"] {{
    background-color: {WARNA_OREN_MUDA} !important;
    border: 1px solid {WARNA_OREN_MUDA} !important;
    color: white !important;
}}
button[kind="primary"]:hover {{
    background-color: {WARNA_OREN_UTAMA} !important;
    border: 1px solid {WARNA_OREN_UTAMA} !important;
    color: white !important;
}}

/* ── CARD PEMBUNGKUS ──────────────────────────────────────────────────
   Semua st.container(border=True, key="vragant_card_...") otomatis dapat
   class CSS "st-key-<key>" dari Streamlit (fitur bawaan sejak Streamlit
   1.37+, ini cara RESMI untuk styling container tertentu -- bukan
   hack lewat data-testid yang gampang berubah tiap update Streamlit).
   Selector wildcard di bawah ini menangkap SEMUA key yang diawali
   "vragant_card", jadi kita cukup tulis 1 CSS untuk semua card
   (form filter maupun hasil rekomendasi) -- tidak perlu diulang manual.
   Kalau versi Streamlit kamu lebih lama dari 1.37 dan belum kenal param
   `key` di st.container, card ini akan tetap tampil (fallback ke kotak
   abu-abu bawaan Streamlit) -- cuma warnanya belum ikut tema. */
div[class*="st-key-vragant_card"] {{
    border: 1.5px solid {WARNA_HIJAU_BORDER} !important;
    border-radius: 18px !important;
    background-color: {WARNA_HIJAU_BG};
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
}}

/* Card produk individual (tampilkan_kartu_produk di ui_components.py)
   dibuat sedikit lebih rapat & rounded juga supaya senada, tanpa perlu
   ubah kode di ui_components.py -- cukup styling lewat class bawaan
   Streamlit untuk SEMUA st.container(border=True) TANPA key khusus. */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px;
}}

/* ── RESPONSIVE / MOBILE ──────────────────────────────────────────────
   layout="centered" di st.set_page_config sudah membuat Streamlit
   otomatis men-stack kolom (st.columns) jadi 1 kolom penuh di layar
   sempit (di bawah ~640px) -- ini perilaku bawaan framework, bukan
   sesuatu yang perlu kita program manual. Kita cuma perlu pastikan
   konten di dalamnya (judul, gambar) ikut menyesuaikan ukuran teks
   supaya tidak kepotong/kekecilan di HP. */
@media (max-width: 640px) {{
    .vragant-judul-brand {{ font-size: 1.6rem !important; }}
    .vragant-tagline {{ font-size: 0.85rem !important; }}
    div[class*="st-key-vragant_card"] {{ padding: 1.25rem 1rem; }}
}}
</style>
"""


def terapkan_tema_cerah():
    """Panggil sekali di awal app.py untuk menerapkan CSS tema."""
    st.markdown(CSS_TEMA_CERAH, unsafe_allow_html=True)


def _logo_ke_base64(path_logo: str) -> str | None:
    """Baca file logo dan ubah jadi base64, supaya bisa ditaruh sebagai
    <img> INLINE di dalam 1 blok HTML yang sama dengan teks judul.

    Kenapa base64, bukan st.image() biasa?
    st.image() selalu jadi elemen block-nya SENDIRI (baris baru sendiri),
    tidak bisa disandingkan sejajar dengan teks lain dalam 1 baris kecuali
    dipaksa lewat st.columns() -- dan itu yang kemarin bikin logo jadi
    segede kolom (`use_container_width=True` ngikutin lebar kolom, bukan
    ukuran asli logo). Dengan base64 di dalam <img> HTML, kita bisa atur
    tinggi logo secara presisi (mis. 56px) dan taruh sejajar teks pakai
    flexbox biasa -- persis seperti logo+wordmark di kebanyakan web.

    Return None kalau file tidak ada/gagal dibaca (supaya pemanggil bisa
    fallback ke slot placeholder).
    """
    try:
        with open(path_logo, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ekstensi = path_logo.rsplit(".", 1)[-1].lower()
        mime = "svg+xml" if ekstensi == "svg" else ekstensi
        return f"data:image/{mime};base64,{data}"
    except Exception:
        return None


def tampilkan_header(path_logo: str | None = None):
    """Tampilkan header: logo kecil SEJAJAR di sebelah kiri judul "VRAGANT AI"
    (bukan ditumpuk di atas), tapi grup logo+judul ini di-CENTER sebagai
    satu kesatuan di tengah halaman.

    Args:
        path_logo: path ke file logo (svg/png). Kalau None atau file
                   tidak ditemukan, slot logo ditampilkan KOSONG (placeholder
                   kotak putus-putus kecil) -- supaya kamu tahu di mana harus
                   menaruh logo kamu sendiri nanti, tanpa app jadi error.
    """
    logo_data_uri = _logo_ke_base64(path_logo) if path_logo else None

    if logo_data_uri:
        logo_html = (
            f"<img src='{logo_data_uri}' "
            "style='height:56px;width:auto;flex-shrink:0;' />"
        )
    else:
        logo_html = (
            "<div style='width:56px;height:56px;border:2px dashed #F0997B;"
            "border-radius:10px;display:flex;align-items:center;"
            "justify-content:center;font-size:9px;color:#993C1D;"
            "text-align:center;flex-shrink:0;line-height:1.1;'>Logo<br>di sini</div>"
        )

    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "gap:14px;flex-wrap:wrap;'>"
        f"{logo_html}"
        "<div style='line-height:1.2;text-align:left;'>"
        "<span class='vragant-judul-brand' style='font-size:2rem;font-weight:700;"
        "font-family:Georgia,serif;color:#4A3B47;'>VRAGANT AI</span><br>"
        "<span class='vragant-tagline' style='font-size:0.95rem;font-style:italic;"
        f"color:{WARNA_OREN_UTAMA};'>Virtual Fragrance Consultant</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center;color:#6b6b6b;font-size:0.9rem;"
        "margin-top:0.5rem;'>"
        "Platform Rekomendasi &amp; Scanner Parfum Lokal Indonesia</p>",
        unsafe_allow_html=True,
    )


def tampilkan_footer(instagram_handle: str = "sayapasha_"):
    """Tampilkan footer standar di paling bawah halaman: nama brand,
    tahun berjalan, dan ajakan kerja sama buat pemilik toko yang mau
    parfumnya ditampilkan di VRAGANT -- link-nya langsung ke profil
    Instagram (bukan cuma teks/tulisan biasa).

    Dipanggil SEKALI saja di app.py, di luar semua tab (st.tabs), supaya
    tampil di bawah tab manapun yang sedang dibuka -- bukan berulang
    tiap tab.

    Args:
        instagram_handle: username IG TANPA "@" (mis. "sayapasha_").
                           Dipisah jadi parameter (bukan ditulis manual di
                           HTML) supaya gampang diganti kalau username
                           berubah, tanpa perlu utak-atik markup-nya.
    """
    tahun_sekarang = datetime.date.today().year
    link_ig = f"https://instagram.com/{instagram_handle}"

    st.divider()
    st.markdown(
        "<div style='text-align:center;padding:0.5rem 0 1.5rem 0;'>"
        "<p style='font-family:Georgia,serif;font-weight:700;font-size:1.15rem;"
        "color:#4A3B47;margin-bottom:0.2rem;'>VRAGANT</p>"
        "<p style='color:#8a8a8a;font-size:0.85rem;margin-bottom:0.9rem;'>"
        f"&copy; {tahun_sekarang} VRAGANT. Semua hak cipta dilindungi.</p>"
        "<p style='font-size:0.9rem;color:#4A3B47;'>"
        "Ingin menampilkan toko Anda dalam website rekomendasi ini? "
        f"Silahkan hubungi IG: <a href='{link_ig}' target='_blank' "
        f"style='color:{WARNA_OREN_UTAMA};font-weight:600;text-decoration:none;'>"
        f"@{instagram_handle}</a>"
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )