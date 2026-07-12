import pandas as pd
import streamlit as st

from core.gemini_client import get_gemini_client
from utils.data_loader import load_catalog, format_harga


def _filter_harga(df: pd.DataFrame, harga_min: int | None, harga_max: int | None) -> pd.DataFrame:
    """Filter katalog berdasarkan range harga user (hard constraint).

    Sama seperti filter gender: ini aturan PASTI, bukan penalaran --
    kalau user bilang budget maksimal 200rb, produk 500rb memang tidak
    relevan, titik. Tidak perlu Gemini "mempertimbangkan" ini, cukup
    filter langsung di kode sebelum data dikirim ke prompt.

    harga_min/harga_max bisa None (user tidak mengisi salah satu/kedua
    batas), yang berarti TIDAK ADA batas di sisi itu.
    """
    if harga_min is not None:
        df = df[df["harga"] >= harga_min]
    if harga_max is not None:
        df = df[df["harga"] <= harga_max]
    return df


def _ringkas_katalog_untuk_prompt(df: pd.DataFrame) -> str:
    """Ubah seluruh katalog (yang sudah difilter) jadi teks ringkas yang
    dikirim di dalam prompt.

    Kenapa diringkas (bukan dump seluruh kolom CSV mentah-mentah)?
    - Mengurangi jumlah token yang dikirim ke API (lebih hemat & lebih cepat).
      Ini makin penting di dataset baru yang isinya 492 produk, bukan 15 --
      filter harga & gender di atas membantu memperkecil katalog sebelum
      dikirim, supaya prompt tidak membengkak.
    - Kolom seperti link_toko atau foto_url tidak relevan untuk PENALARAN
      Gemini -- itu cuma dibutuhkan nanti untuk TAMPILAN, jadi tidak perlu
      ikut dikirim sebagai konteks reasoning.
    """
    baris = []
    for _, row in df.iterrows():
        baris.append(
            f"- id_produk={row['id_produk']} | {row['nama_parfum']} ({row['brand']}) | "
            f"harga={format_harga(row['harga'])} | "
            f"gender={row['gender']} | kategori={row['kategori']} | "
            f"notes: {row['top_notes']}, {row['middle_notes']}, {row['base_notes']} | "
            f"kesan: {row['deskripsi_wangi']} | "
            f"cocok untuk: {row['aktivitas']}"
        )
    return "\n".join(baris)


def _susun_prompt(
    aktivitas: list[str],
    waktu: list[str],
    gender: str,
    lokasi: str,
    deskripsi_custom: str,
    katalog_teks: str,
    jumlah_rekomendasi: int,
) -> str:
    """Susun prompt lengkap untuk Gemini.

    Prinsip prompt engineering yang dipakai:
    - Beri PERAN yang jelas ("kamu adalah konsultan parfum ahli")
    - Beri KONTEKS lengkap user dalam bahasa natural, bukan kode/angka
    - Jelaskan bagaimana menangani konteks MAJEMUK (multi-aktivitas/waktu)
      secara eksplisit, supaya Gemini tahu harus mencari parfum yang
      "menyambung" antar konteks, bukan asal pilih salah satu
    - Kalau user menulis deskripsi wangi sendiri (opsional), beri itu
      BOBOT LEBIH (bukan menggantikan pilihan lain), karena itu sinyal
      paling eksplisit/personal dari keinginan user
    - Minta output JSON dengan skema yang didefinisikan jelas (field apa
      saja, tipe apa) supaya hasilnya konsisten dan mudah diparsing
    - jumlah_rekomendasi dibuat sebagai PARAMETER (bukan angka hardcode di
      teks prompt) supaya gampang diubah dari satu tempat saja kalau nanti
      mau ditambah/dikurangi lagi -- contoh konkret kenapa "magic number"
      sebaiknya dihindari dalam kode.
    """
    daftar_aktivitas = ", ".join(aktivitas) if aktivitas else "tidak ditentukan"
    daftar_waktu = ", ".join(waktu) if waktu else "tidak ditentukan"

    catatan_majemuk = ""
    if len(aktivitas) > 1 or len(waktu) > 1:
        catatan_majemuk = (
            "\nPENTING: User memilih LEBIH DARI SATU aktivitas/waktu sekaligus "
            "(artinya dia akan menjalani semuanya di hari yang sama tanpa ganti "
            "parfum, misalnya kuliah pagi lanjut nongkrong sore). Carikan parfum "
            "yang tetap cocok dan tidak terasa aneh di SEMUA konteks yang dipilih, "
            "bukan cuma cocok di salah satu konteks saja."
        )

    catatan_custom = ""
    if deskripsi_custom and deskripsi_custom.strip():
        catatan_custom = (
            f"\nPREFERENSI WANGI SPESIFIK DARI USER (sangat penting, beri "
            f"bobot lebih dibanding konteks aktivitas/waktu di atas karena "
            f"ini permintaan personal yang eksplisit):\n\"{deskripsi_custom.strip()}\""
        )

    contoh_json = ",\n    ".join(
        f'{{"id_produk": {i}, "alasan": "Cocok karena ..."}}' for i in [6, 12, 3]
    )

    return f"""Kamu adalah konsultan parfum ahli yang membantu pengguna memilih
parfum lokal Indonesia yang paling sesuai dengan kebutuhan mereka hari ini.

KONTEKS USER:
- Aktivitas: {daftar_aktivitas}
- Waktu: {daftar_waktu}
- Gender: {gender}
- Lokasi: {lokasi}
{catatan_majemuk}{catatan_custom}

KATALOG PARFUM YANG TERSEDIA (sudah difilter sesuai gender & budget user):
{katalog_teks}

TUGAS:
Pilih TEPAT {jumlah_rekomendasi} parfum dari katalog di atas yang paling
sesuai dengan konteks user, diurutkan dari yang PALING cocok ke yang
KURANG cocok (tapi tetap layak masuk daftar). Untuk masing-masing, beri
alasan singkat (1-2 kalimat, bahasa santai, sebutkan kenapa cocok dengan
konteks spesifik user di atas -- bukan penjelasan generik soal aromanya
saja). Kalau katalog yang tersedia kurang dari {jumlah_rekomendasi}, pilih
SEMUA yang tersedia saja.

Balas HANYA dalam format JSON seperti ini, tanpa teks lain:
{{
  "rekomendasi": [
    {contoh_json}
  ]
}}"""


def cari_rekomendasi_berdasarkan_aktivitas(
    aktivitas: list[str],
    waktu: list[str],
    gender: str,
    lokasi: str,
    deskripsi_custom: str = "",
    harga_min: int | None = None,
    harga_max: int | None = None,
    jumlah_rekomendasi: int = 5,
) -> pd.DataFrame:
    """Fungsi utama yang dipanggil dari Tab 1 di app.py.

    Args:
        aktivitas, waktu, gender, lokasi: konteks dasar (sudah ada sejak awal).
        deskripsi_custom: kalimat bebas opsional dari user soal wangi yang
                           diinginkan. String kosong = user tidak mengisi.
        harga_min, harga_max: batas budget user dalam Rupiah (angka, bukan
                               teks). None = tidak ada batas di sisi itu.
        jumlah_rekomendasi: berapa banyak produk yang diminta dari Gemini.
                             Dikontrol langsung oleh user lewat input di
                             app.py (maksimal 10, supaya Gemini tidak butuh
                             waktu terlalu lama menyusun alasan untuk
                             tiap rekomendasi). Default 5 di sini hanya
                             dipakai kalau fungsi ini dipanggil tanpa
                             lewat UI (misal saat testing).

    Returns:
        DataFrame berisi maksimal jumlah_rekomendasi produk, dengan kolom
        tambahan "alasan_gemini" yang berisi penjelasan dari Gemini.
        DataFrame kosong kalau terjadi error (sudah ditangani dengan
        pesan st.warning/st.error di dalam, sesuai pola gemini_client.py).
    """
    client = get_gemini_client()
    df = load_catalog()

    df_filtered = df[df["gender"] == gender]

    df_filtered = _filter_harga(df_filtered, harga_min, harga_max)

    if df_filtered.empty:
        st.warning(
            "Tidak ada produk yang cocok dengan filter gender dan/atau "
            "range harga ini. Coba lebarkan budget atau ubah filter."
        )
        return pd.DataFrame()

    katalog_teks = _ringkas_katalog_untuk_prompt(df_filtered)
    prompt = _susun_prompt(
        aktivitas, waktu, gender, lokasi, deskripsi_custom, katalog_teks,
        jumlah_rekomendasi=jumlah_rekomendasi,
    )

    hasil_json = client.generate_json(prompt)

    if not hasil_json or "rekomendasi" not in hasil_json:
        st.warning("Gemini tidak memberikan rekomendasi yang valid, coba lagi.")
        return pd.DataFrame()

    baris_hasil = []
    for item in hasil_json["rekomendasi"]:
        id_produk = item.get("id_produk")
        alasan = item.get("alasan", "")
        match = df[df["id_produk"] == id_produk]
        if match.empty:
            continue
        row = match.iloc[0].copy()
        row["alasan_gemini"] = alasan
        baris_hasil.append(row)

    if not baris_hasil:
        st.warning(
            "Gemini memberikan id_produk yang tidak ditemukan di katalog. "
            "Coba lagi."
        )
        return pd.DataFrame()

    return pd.DataFrame(baris_hasil).reset_index(drop=True)