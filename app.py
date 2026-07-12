"""
app.py
------
Entry point Streamlit VRAGANT AI dengan 3 tab:
- Tab 1 ScentDaily: rekomendasi berdasarkan aktivitas
  (2 mode: Lokal TF-IDF atau AI Gemini)
- Tab 2 ScentNote: rekomendasi berdasarkan preferensi aroma
  (2 mode: Lokal TF-IDF atau AI Gemini embedding)
- Tab 3 Tara: chatbot konsultan parfum berbasis Gemini
"""

import streamlit as st
from core.gemini_client import get_gemini_client
from core.embedding_engine import cari_rekomendasi_berdasarkan_notes
from core.recommender_engine import cari_rekomendasi_berdasarkan_aktivitas
from core.local_engine import (
    cari_lokal_berdasarkan_aktivitas,
    cari_lokal_berdasarkan_notes,
)
from core.chatbot_engine import (
    inisialisasi_chat,
    kirim_pesan,
    PESAN_SELAMAT_DATANG,
)
from utils.ui_components import tampilkan_daftar_produk, tampilkan_kartu_produk
from utils.theme import terapkan_tema_cerah, tampilkan_header, tampilkan_footer

st.set_page_config(page_title="VRAGANT AI", page_icon="✨", layout="centered")

terapkan_tema_cerah()
tampilkan_header(path_logo="assets/branding/logo.png")
st.divider()

tab1, tab2, tab3, tab_tes = st.tabs([
    "ScentDaily", "ScentNote", "Tara Chatbot", "Tes Koneksi"
])


# ──────────────────────────────────────────────────────────────────────
# HELPER: komponen yang dipakai ulang di beberapa tab
# ──────────────────────────────────────────────────────────────────────

def _input_jumlah_rekomendasi(key_prefix: str, default: int = 5) -> int:
    return st.number_input(
        "Jumlah rekomendasi (maks. 10)",
        min_value=1, max_value=10, value=default, step=1,
        key=f"{key_prefix}_jumlah",
        help="Dibatasi 10 supaya tidak terlalu berat di mode AI.",
    )


def _input_range_harga(key_prefix: str):
    st.caption("Filter budget (opsional, biarkan 0 kalau tidak ingin membatasi)")
    col_min, col_max = st.columns(2)
    with col_min:
        hmin = st.number_input("Harga minimal (Rp)", min_value=0, value=0,
                               step=10_000, key=f"{key_prefix}_hmin")
    with col_max:
        hmax = st.number_input("Harga maksimal (Rp)", min_value=0, value=0,
                               step=10_000, key=f"{key_prefix}_hmax")
    return (hmin if hmin > 0 else None), (hmax if hmax > 0 else None)


# ──────────────────────────────────────────────────────────────────────
# TAB 1 — SCENTDAILY
# ──────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Rekomendasi parfum berdasarkan aktivitas")
    st.caption(
        "Pilih konteks aktivitasmu hari ini. Tersedia dua mode: "
        "**Lokal** (cepat, offline, tanpa API) dan **AI Gemini** (lebih cerdas, butuh koneksi)."
    )

    # CARD 1: form input/filter. key harus diawali "vragant_card" supaya
    # kena styling CSS di theme.py (lihat CSS_TEMA_CERAH). key juga harus
    # unik antar container (makanya diberi suffix "_scentdaily").
    with st.container(border=True, key="vragant_card_form_scentdaily"):
        aktivitas = st.multiselect(
            "Aktivitas (boleh lebih dari satu)",
            ["Sekolah", "Kuliah", "Kantor", "Ngedate", "Nongkrong", "Olahraga", "Acara Formal"],
            default=[],
        )
        waktu = st.multiselect(
            "Waktu (boleh lebih dari satu)",
            ["Pagi", "Siang", "Malam"],
            default=[],
        )

        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("Gender", ["male", "female", "unisex"], horizontal=True)
        with col2:
            lokasi = st.radio("Lokasi", ["indoor", "outdoor"], horizontal=True)

        deskripsi_custom = st.text_area(
            "Deskripsi wangi yang kamu inginkan (opsional)",
            placeholder="Contoh: aku mau yang seger kayak abis mandi, gak terlalu manis.",
            help="Dipakai di kedua mode (lokal maupun AI Gemini).",
        )

        harga_min_t1, harga_max_t1 = _input_range_harga("tab1")
        jumlah_t1 = _input_jumlah_rekomendasi("tab1")

        # Dua tombol sejajar: Lokal dan AI Gemini
        col_btn1, col_btn2 = st.columns(2)
        klik_lokal_t1 = col_btn1.button(
            "🔍 Cari (Lokal)", key="btn_lokal_t1",
            help="Menggunakan model TF-IDF yang berjalan di lokal, tanpa internet.",
            use_container_width=True,
        )
        klik_ai_t1 = col_btn2.button(
            "✨ Cari (AI Gemini)", key="btn_ai_t1",
            type="primary",
            help="Menggunakan Gemini AI, butuh koneksi internet dan kuota API.",
            use_container_width=True,
        )

    if klik_lokal_t1 or klik_ai_t1:
        if not aktivitas or not waktu:
            st.warning("Pilih minimal 1 aktivitas dan 1 waktu dulu ya.")
        else:
            if klik_lokal_t1:
                with st.spinner("Menghitung rekomendasi lokal..."):
                    hasil = cari_lokal_berdasarkan_aktivitas(
                        aktivitas, waktu, gender, lokasi,
                        deskripsi_custom=deskripsi_custom,
                        harga_min=harga_min_t1, harga_max=harga_max_t1,
                        jumlah_rekomendasi=jumlah_t1,
                    )
                # CARD 2: hasil rekomendasi -- semua kartu produk dibungkus
                # 1 card besar yang sama, bukan card terpisah per hasil.
                with st.container(border=True, key="vragant_card_hasil_scentdaily"):
                    if not hasil.empty:
                        st.info("🔍 Hasil dari model **Lokal** (TF-IDF)")
                        tampilkan_daftar_produk(hasil, kolom_skor="skor_lokal")
                    else:
                        st.error("Tidak ada hasil. Coba ubah filter.")

            else:  # klik_ai_t1
                with st.spinner("Gemini sedang menganalisis konteks kamu..."):
                    hasil = cari_rekomendasi_berdasarkan_aktivitas(
                        aktivitas, waktu, gender, lokasi,
                        deskripsi_custom=deskripsi_custom,
                        harga_min=harga_min_t1, harga_max=harga_max_t1,
                        jumlah_rekomendasi=jumlah_t1,
                    )
                with st.container(border=True, key="vragant_card_hasil_scentdaily"):
                    if not hasil.empty:
                        st.info("✨ Hasil dari **AI Gemini**")
                        tampilkan_daftar_produk(hasil, kolom_skor=None)
                        for _, row in hasil.iterrows():
                            st.caption(
                                f"**Kenapa {row['nama_parfum']} cocok:** {row['alasan_gemini']}"
                            )
                    else:
                        st.error("Tidak ada hasil. Cek pesan di atas atau koneksi API.")


# ──────────────────────────────────────────────────────────────────────
# TAB 2 — SCENTNOTE
# ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Cari parfum berdasarkan preferensi aroma")
    st.caption(
        "Atur slider sesuai selera aromamu. Tersedia dua mode: "
        "**Lokal** (offline, tanpa API) dan **AI Gemini** (lebih akurat, tetapi ada limit API)."
    )

    with st.container(border=True, key="vragant_card_form_scentnote"):
        col1, col2 = st.columns(2)
        with col1:
            fresh = st.slider("🍋 Fresh (citrus, green, aquatic)", 1, 5, 3)
            floral = st.slider("🌸 Floral (bunga-bungaan)", 1, 5, 3)
            fruity = st.slider("🍓 Fruity (buah-buahan)", 1, 5, 3)
        with col2:
            woody = st.slider("🪵 Woody (kayu-kayuan)", 1, 5, 3)
            oriental = st.slider("☕ Oriental/Gourmand (rempah, vanila, kopi)", 1, 5, 3)

        harga_min_t2, harga_max_t2 = _input_range_harga("tab2")
        jumlah_t2 = _input_jumlah_rekomendasi("tab2")

        col_btn1, col_btn2 = st.columns(2)
        klik_lokal_t2 = col_btn1.button(
            "🔍 Cari (Lokal)", key="btn_lokal_t2",
            help="TF-IDF lokal, cocokkan kata-kata notes secara langsung.",
            use_container_width=True,
        )
        klik_ai_t2 = col_btn2.button(
            "✨ Cari (AI Gemini)", key="btn_ai_t2",
            type="primary",
            help="Gemini embedding, mengerti makna di balik kata (semantik).",
            use_container_width=True,
        )

    if klik_lokal_t2 or klik_ai_t2:
        skor = {
            "fresh": fresh, "floral": floral, "fruity": fruity,
            "woody": woody, "oriental": oriental,
        }

        if klik_lokal_t2:
            with st.spinner("Menghitung kemiripan lokal..."):
                hasil = cari_lokal_berdasarkan_notes(
                    skor, top_n=jumlah_t2,
                    harga_min=harga_min_t2, harga_max=harga_max_t2,
                )
            with st.container(border=True, key="vragant_card_hasil_scentnote"):
                if not hasil.empty:
                    st.info("🔍 Hasil dari model **Lokal** (TF-IDF)")
                    tampilkan_daftar_produk(hasil, kolom_skor="similarity_score")
                else:
                    st.error("Tidak ada hasil. Coba ubah filter.")

        else:  # klik_ai_t2
            with st.spinner("Menghitung kemiripan semantik dengan Gemini..."):
                hasil = cari_rekomendasi_berdasarkan_notes(
                    skor, top_n=jumlah_t2,
                    harga_min=harga_min_t2, harga_max=harga_max_t2,
                )
            with st.container(border=True, key="vragant_card_hasil_scentnote"):
                if not hasil.empty:
                    st.info("✨ Hasil dari **AI Gemini** (embedding semantik)")
                    tampilkan_daftar_produk(hasil, kolom_skor="similarity_score")
                else:
                    st.error("Tidak ada hasil. Cek pesan di atas atau koneksi API.")


# ──────────────────────────────────────────────────────────────────────
# TAB 3 — TARA CHATBOT
# ──────────────────────────────────────────────────────────────────────
with tab3:
    inisialisasi_chat()

    st.subheader("Tara — Konsultan Parfum Virtual")

    # Tampilkan pesan selamat datang sekali di awal
    if not st.session_state.tara_greeted:
        with st.chat_message("assistant", avatar="🌸"):
            st.markdown(PESAN_SELAMAT_DATANG)
        st.session_state.tara_greeted = True

    # Tampilkan riwayat percakapan
    for msg in st.session_state.tara_messages:
        avatar = "🌸" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input dari user
    pesan_user = st.chat_input("Tulis pesanmu ke Tara...")
    if pesan_user:
        # Tampilkan pesan user
        with st.chat_message("user", avatar="🧑"):
            st.markdown(pesan_user)
        st.session_state.tara_messages.append(
            {"role": "user", "content": pesan_user}
        )

        # Dapatkan balasan dari Tara
        with st.chat_message("assistant", avatar="🌸"):
            with st.spinner("Tara sedang berpikir..."):
                try:
                    client = get_gemini_client()
                    balasan = kirim_pesan(pesan_user, client)
                except Exception:
                    balasan = (
                        "Mohon maaf, kuota request habis hari ini 🙏 "
                        "Silahkan coba fitur ScentDaily atau ScentNote ya!"
                    )
            st.markdown(balasan)

        st.session_state.tara_messages.append(
            {"role": "assistant", "content": balasan}
        )

    # Tombol reset percakapan
    if st.session_state.tara_messages:
        if st.button("🔄 Reset percakapan", key="reset_chat"):
            st.session_state.tara_messages = []
            st.session_state.tara_greeted = False
            st.rerun()


# ──────────────────────────────────────────────────────────────────────
# TAB TES KONEKSI
# ──────────────────────────────────────────────────────────────────────
with tab_tes:
    st.subheader("Tes Koneksi Gemini API")
    st.caption("Klik tombol di bawah untuk memastikan API key sudah terpasang dengan benar.")
    if st.button("Tes koneksi ke Gemini"):
        with st.spinner("Menghubungi Gemini..."):
            try:
                client = get_gemini_client()
                hasil_tes = client.generate_text(
                    "Balas dengan satu kalimat singkat: koneksi berhasil."
                )
                if hasil_tes:
                    st.success(f"✅ Gemini menjawab: {hasil_tes}")
                else:
                    st.error("Tidak ada respon. Cek API key di .streamlit/secrets.toml")
            except Exception as e:
                st.error(f"Gagal terhubung: {e}")


# ──────────────────────────────────────────────────────────────────────
# FOOTER -- di luar semua tab, jadi tampil di bawah halaman apa pun yang
# sedang dibuka user.
# ──────────────────────────────────────────────────────────────────────
tampilkan_footer(instagram_handle="sayapasha_")