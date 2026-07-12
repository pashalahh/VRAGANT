"""
core/chatbot_engine.py
-----------------------
Logika chatbot "Tara" -- konsultan parfum virtual berbasis Gemini.

Tara adalah karakter AI yang membantu pengguna berdiskusi bebas tentang
parfum: rekomendasi informal, penjelasan istilah (top/middle/base notes,
famili aroma), atau sekadar curhat soal selera wangi -- di luar format
form terstruktur ScentDaily/ScentNote yang lebih formal.

ARSITEKTUR:
- Multi-turn conversation: Tara ingat konteks percakapan dalam satu sesi
  (disimpan di st.session_state, bukan database -- jadi direset kalau
  halaman di-refresh, ini disengaja untuk privasi & kesederhanaan).
- System prompt: Tara diberi "kepribadian" dan konteks katalog ringkas
  lewat system prompt, supaya jawabannya tetap relevan ke parfum lokal
  Indonesia, bukan menjawab pertanyaan umum di luar topik.
- Error handling 429: kalau kuota Gemini habis, tampilkan pesan ramah
  bukan error mentah -- ini jauh lebih baik dari sisi UX, terutama
  saat demo/presentasi.
"""

import streamlit as st


SYSTEM_PROMPT = """Kamu adalah Tara, konsultan parfum virtual dari platform VRAGANT AI.
Kamu membantu pengguna menemukan dan memahami parfum lokal Indonesia.

Kepribadianmu:
- Ramah, santai, dan informatif -- seperti teman yang kebetulan ahli parfum
- Menjawab dalam Bahasa Indonesia yang natural (boleh campur sedikit Inggris kalau perlu istilah teknis)
- Fokus ke parfum lokal Indonesia, tapi boleh referensikan parfum internasional sebagai perbandingan
- Kalau ditanya rekomendasi, berikan 2-3 saran konkret dengan alasan singkat
- Kalau tidak tahu jawabannya, jujur dan sarankan user mencoba fitur ScentDaily atau ScentNote

Hal yang kamu TIDAK lakukan:
- Menjawab pertanyaan di luar topik parfum/kecantikan/lifestyle yang berkaitan
- Memberikan informasi medis atau klaim kesehatan terkait wewangian
- Berpura-pura menjadi AI lain atau karakter lain
"""

PESAN_SELAMAT_DATANG = (
    "Halo! Saya Tara, konsultan fragrance virtual kamu 🌸 "
    "Apa yang bisa saya bantu hari ini? "
    "Mau cari parfum buat acara tertentu, atau penasaran soal notes & aroma tertentu?"
)

PESAN_KUOTA_HABIS = (
    "Mohon maaf, kuota request AI kami sudah habis untuk hari ini 🙏 "
    "Silahkan coba fitur **ScentDaily** atau **ScentNote** yang tetap tersedia, "
    "atau kembali lagi besok ya!"
)


def inisialisasi_chat():
    """Siapkan state awal percakapan kalau belum ada.

    Dipanggil sekali di awal tab Tara di app.py. Menggunakan
    st.session_state supaya riwayat chat tetap ada selama user
    berada di halaman yang sama (tapi direset kalau refresh --
    ini disengaja, bukan bug).

    Format messages mengikuti standar OpenAI/Gemini:
    [{"role": "user"/"assistant", "content": "..."}]
    """
    if "tara_messages" not in st.session_state:
        st.session_state.tara_messages = []
    if "tara_greeted" not in st.session_state:
        st.session_state.tara_greeted = False


def kirim_pesan(pesan_user: str, client) -> str:
    """Kirim pesan ke Gemini dengan konteks percakapan penuh, terima balasan.

    Args:
        pesan_user: teks yang diketik user di chat input.
        client: GeminiClient instance dari get_gemini_client().

    Returns:
        String balasan dari Tara. Kalau kena rate limit (429), kembalikan
        PESAN_KUOTA_HABIS (bukan error mentah).
    """
    # Susun history percakapan untuk dikirim ke Gemini
    contents = []
    for msg in st.session_state.tara_messages:
        contents.append({
            "role": msg["role"],
            "parts": [{"text": msg["content"]}]
        })
    # Tambahkan pesan user yang baru
    contents.append({
        "role": "user",
        "parts": [{"text": pesan_user}]
    })

    try:
        from google.genai import types
        response = client._client.models.generate_content(
            model=client.TEXT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,  # sedikit lebih tinggi dari ScentDaily
                                  # supaya percakapan terasa lebih natural/variatif
            ),
        )
        return response.text

    except Exception as e:
        err = str(e)
        # Deteksi error 429 (kuota habis) -- tampilkan pesan ramah,
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return PESAN_KUOTA_HABIS
        # Error lain (koneksi, model, dll) -- pesan generic yang tetap ramah
        return (
            "Waduh, ada gangguan teknis sementara nih 😅 "
            "Coba kirim pesannya lagi ya, atau cek koneksi internet kamu."
        )
