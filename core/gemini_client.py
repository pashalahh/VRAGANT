import json
import time
import streamlit as st
from google import genai
from google.genai import types
from google.genai import errors as genai_errors


class GeminiClient:
    """Pembungkus tipis di atas Google GenAI SDK (SDK baru, bukan yang deprecated)."""

    TEXT_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "gemini-embedding-001"

    MAKS_RETRY_RATE_LIMIT = 3
    JEDA_AWAL_DETIK = 12

    def __init__(self):

        api_key = st.secrets.get("GEMINI_API_KEY")

        if not api_key or api_key.startswith("isi-api-key"):
            st.error(
                "GEMINI_API_KEY belum diisi. Buka .streamlit/secrets.toml "
                "dan isi dengan API key asli kamu dari Google AI Studio."
            )
            st.stop()

        self._client = genai.Client(api_key=api_key)

    def _adalah_error_rate_limit(self, error: Exception) -> bool:
        """Cek apakah sebuah error adalah error 429 (kuota/rate limit habis),
        BUKAN error lain (API key salah, model tidak ada, dll).

        PENTING bedanya: rate limit itu SEMENTARA (akan reset sendiri dalam
        hitungan detik), sedangkan error lain (401 API key salah, 404 model
        tidak ditemukan) PERMANEN -- mengulang-ulang request yang salah
        konfigurasi cuma buang waktu, jadi retry HANYA dilakukan kalau
        errornya benar-benar 429.
        """
        if isinstance(error, genai_errors.ClientError):
            return getattr(error, "code", None) == 429

        return "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)

    def _retry_dengan_backoff(self, fungsi_panggilan, label: str):
        """Jalankan fungsi_panggilan(), dan kalau gagal karena rate limit (429),
        tunggu sebentar lalu coba lagi -- maksimal MAKS_RETRY_RATE_LIMIT kali.

        Dipusatkan di sini (bukan ditulis ulang di embed_text DAN embed_batch
        secara terpisah) supaya logika retry konsisten dan mudah diubah dari
        satu tempat saja kalau diperlukan.
        """
        jeda = self.JEDA_AWAL_DETIK
        for percobaan in range(1, self.MAKS_RETRY_RATE_LIMIT + 1):
            try:
                return fungsi_panggilan()
            except Exception as e:
                if not self._adalah_error_rate_limit(e) or percobaan == self.MAKS_RETRY_RATE_LIMIT:
                    raise  # bukan rate limit, ATAU sudah kehabisan jatah retry -> lempar ke pemanggil
                st.warning(
                    f"{label}: kuota API sedang penuh (percobaan {percobaan}/"
                    f"{self.MAKS_RETRY_RATE_LIMIT}), mencoba lagi dalam {jeda} detik..."
                )
                time.sleep(jeda)
                jeda *= 2  # exponential backoff: jeda berikutnya 2x lebih lama

    def generate_text(self, prompt: str) -> str:
        """Kirim prompt teks biasa, terima jawaban teks biasa.
        Dipakai misalnya untuk fallback atau penjelasan singkat."""
        try:
            response = self._client.models.generate_content(
                model=self.TEXT_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            st.warning(f"Gemini sedang tidak bisa diakses: {e}")
            return ""

    def generate_json(self, prompt: str) -> dict:
        """Minta Gemini menjawab dalam format JSON terstruktur, lalu di-parse jadi dict."""
        try:
            response = self._client.models.generate_content(
                model=self.TEXT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4,
                ),
            )
            return json.loads(response.text)
        except json.JSONDecodeError:
            st.warning("Gemini tidak mengembalikan JSON yang valid, coba lagi.")
            return {}
        except Exception as e:
            if self._adalah_error_rate_limit(e):
                st.warning(
                    "⚠️ Mohon maaf, kuota request AI kami sudah habis untuk saat ini. "
                    "Silahkan coba **tombol Lokal** di bawah yang tetap tersedia tanpa internet, "
                    "atau kembali lagi beberapa saat lagi ya!"
                )
            else:
                st.warning(f"Gemini sedang tidak bisa diakses: {e}")
            return {}

    def generate_with_image(self, prompt: str, image) -> str:
        """Kirim prompt + gambar (PIL.Image), terima jawaban teks."""
        try:
            response = self._client.models.generate_content(
                model=self.TEXT_MODEL,
                contents=[prompt, image],
            )
            return response.text
        except Exception as e:
            st.warning(f"Gemini Vision sedang tidak bisa diakses: {e}")
            return ""

    def embed_text(self, text: str) -> list[float]:
        """Ubah teks jadi vector embedding (list angka)."""
        try:
            result = self._retry_dengan_backoff(
                lambda: self._client.models.embed_content(
                    model=self.EMBEDDING_MODEL,
                    contents=text,
                ),
                label="Membuat embedding",
            )
            return result.embeddings[0].values
        except Exception as e:
            if self._adalah_error_rate_limit(e):
                st.warning(
                    "⚠️ Mohon maaf, kuota request AI kami sudah habis untuk saat ini. "
                    "Silahkan coba **tombol Lokal** di bawah yang tetap tersedia tanpa internet, "
                    "atau kembali lagi beberapa saat lagi ya!"
                )
            else:
                st.warning(f"Gagal membuat embedding: {e}")
            return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Versi batch dari embed_text, untuk precompute embedding seluruh katalog."""
        try:
            result = self._retry_dengan_backoff(
                lambda: self._client.models.embed_content(
                    model=self.EMBEDDING_MODEL,
                    contents=texts,
                ),
                label="Membuat embedding batch",
            )
            return [emb.values for emb in result.embeddings]
        except Exception as e:
            if self._adalah_error_rate_limit(e):
                st.warning(
                    "⚠️ Mohon maaf, kuota request AI kami sudah habis untuk saat ini. "
                    "Silahkan coba **tombol Lokal** di bawah yang tetap tersedia tanpa internet, "
                    "atau kembali lagi beberapa saat lagi ya!"
                )
            else:
                st.warning(f"Gagal membuat embedding batch: {e}")
            return []


@st.cache_resource
def get_gemini_client() -> GeminiClient:
    """Pastikan GeminiClient hanya dibuat SEKALI per sesi, bukan setiap rerun.

    Streamlit menjalankan ulang seluruh script tiap ada interaksi user
    (klik tombol, geser slider, dst). Tanpa cache ini, kita akan
    re-konfigurasi SDK setiap kali -- boros dan tidak perlu.
    """
    return GeminiClient()