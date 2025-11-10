# Council of Consciousness

Dokumentasi arsitektur dan tata cara penggunaan modul **Council of Consciousness** pada proyek ini. Pendekatan ini memodelkan “parlemen AI” yang menampung berbagai arketipe intelektual untuk menghasilkan pemahaman kolektif yang kaya konteks dan terdokumentasi rapi.

---

## 1. Gambaran Umum

- **Tujuan**: Membuat diskusi multi-agen yang terstruktur, dengan fokus pada kolaborasi, refleksi, dan narasi seimbang.
- **Fitur utama**:
  - Streaming output real-time di terminal (dengan status fase).
  - Logger otomatis ke Markdown.
  - Episodic memory (SQLite) + semantic recall (embeddings `granite-embedding:latest`).
  - Archetype signature voice untuk tiap agen.
  - Evaluasi eliminasi opsional di akhir sesi refleksi.

---

## 2. Arsitektur Sistem

### 2.1 Komponen Utama

| Komponen | Berkas | Fungsi |
| --- | --- | --- |
| Roles registry | `council/roles.py` | Definisi arketipe, model Ollama, dan gaya bicara. |
| Council orchestrator | `council/consciousness.py` | Mengelola loop diskusi: pembukaan → argumen → kritik → refleksi → penutupan (+ eliminasi). |
| Memory subsystem | `council/memory.py` | SQLite episodic memory, embedding, dan summarizer. |
| Interactive wizard | `council/interactive.py` | Antarmuka terminal untuk memilih mode (debate/council) dengan opsi lengkap. |
| CLI entrypoints | `council/cli.py` | Perintah `consciousness`, `interactive`, dan `debate`. |

### 2.2 Alur Diskusi

1. **Pembukaan Moderator**  
   Moderator membuka sesi, menyiapkan konteks, dan menyebutkan fokus diskusi.

2. **Putaran Argumen**  
   Setiap arketipe (rasionalis, humanis, spiritualis, teknokrat, dll.) memberi argumen awal berdasarkan memori + riwayat dialog.

3. **Sesi Kritik & Sanggahan**  
   Kritikus Radikal mengevaluasi logika, bias, dan struktur kekuasaan.

4. **Refleksi Kolektif**  
   Semua agen diberi kesempatan merevisi atau memperkuat posisinya setelah mendengar kritik.

5. **Penutupan Moderator**  
   Moderator merangkum, menilai konsensus/narasi akhir, dan menetapkan tindak lanjut (jika perlu).

6. **Evaluasi Eliminasi (Opsional)**  
   Evaluator otomatis merekomendasikan peran yang kontribusinya paling lemah—berguna untuk sesi lanjutan.

Setiap fase memiliki indikator `Fase x/y` di terminal dan heading Markdown yang sesuai.

---

## 3. Arketipe dan Personalitas

| Role | Model | Warna | Signature Voice |
| --- | --- | --- | --- |
| Moderator (Chair) | `gemma3:1b` | putih | Pertanyaan eksploratif, bahasa formal, fokus merangkum. |
| Filosof Rasionalis | `qwen2.5:3b` | cyan | Argumen logis berstruktur (premis → konsekuensi → kesimpulan). |
| Humanis Empatik | `gemma3:latest` | magenta | Narasi empatik, kisah manusia, fokus kesejahteraan sosial. |
| Kritikus Radikal | `qwen3:1.7b` | kuning | Menantang status quo, mengungkap bias struktural. |
| Spiritualis Mistik | `gemma3:1b` | hijau | Kontemplatif, metafora spiritual, keseimbangan batin. |
| Teknokratis AI | `qwen3:1.7b` | biru terang | Visioner, berbasis data, menyoroti roadmap teknologi. |

Semua suara diselaraskan dengan parameter `reasoning_depth` dan `truth_seeking` agar konsisten.

---

## 4. Memori & Konteks

### 4.1 Episodic Memory (SQLite)

- Lokasi: `memory/council_memory.db`
- Kolom kunci: `question`, `agent`, `role`, `phase`, `content`, `embedding`.
- Dicatat untuk setiap fase (opening, argument, critique, reflection, closing, elimination).

### 4.2 Semantic Recall

- Embedding: `granite-embedding:latest` melalui `langfuse.openai` client.
- Fungsi `fetch_similar(...)` pada `CouncilMemory` memanggil memori relevan (cosine similarity).
- Ringkasan memori lama dibuat dengan model `gemma3:1b` dan diberikan ke setiap agen di awal fase.

### 4.3 Markdown Log

- Lokasi: `debates/<timestamp>_<judul>_council.md`
- Berisi heading fase + kontribusi setiap agen.
- Mudah dibaca ulang atau dibagikan sebagai arsip diskusi.

---

## 5. Cara Menjalankan

### 5.1 Prasyarat

Pastikan model Ollama berikut telah diunduh:

```bash
ollama pull gemma3:1b
ollama pull gemma3:latest
ollama pull qwen2.5:3b
ollama pull qwen3:1.7b
ollama pull granite-embedding:latest
```

### 5.2 Menjalankan Council of Consciousness

```bash
uv run -m council.cli consciousness --question "Apakah AI dapat menjadi pemimpin moral umat manusia?"
# Opsi tambahan:
#   --title "Sesi 1"
#   --eliminate     # aktifkan rekomendasi eliminasi
```

### 5.3 Wizard Interaktif

```bash
uv run -m council.cli interactive
```

Di wizard:
- Pilih mode `council` atau `debate`.
- Masukkan pertanyaan, judul, jumlah iterasi (jika debate), preset konsensus, dan toggle eliminasi.
- Pilih agen dengan memasukkan nomor atau `all`.

---

## 6. Integrasi Observability

- Menggunakan `langfuse.openai.OpenAI` sehingga seluruh panggilan chat + embeddings otomatis ditrace ke Langfuse.
- Kunci API berada di `.env` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`).
- Endpoint Ollama `http://localhost:11434/v1` dipakai sebagai `base_url`.

---

## 7. Pengembangan Lanjutan

Ide tambahan:

1. **Hierarchical Delegation**  
   Tambah agent supervisor yang memutus kapan memanggil arketipe tertentu.

2. **Parallel Batch Rounds**  
   Menjalankan beberapa arketipe secara paralel untuk mempercepat putaran.

3. **Evaluation Metrics**  
   Simpan skor kualitas argumen, pemetaan konsensus, atau sentiment analysis di laporan akhir.

4. **UI/Visualization**  
   Render log Markdown ke dashboard web atau PDF.

5. **Tool-use / Retrieval Injection**  
   Integrasikan RAG ekstra (misalnya Chroma/FAISS) untuk pencarian dokumen domain spesifik.

---

## 8. Struktur Direktori Terkait

```
council/
 ├─ cli.py                 # Entry CLI (debate, interactive, consciousness)
 ├─ consciousness.py       # Orkestrator Council of Consciousness
 ├─ interactive.py         # Wizard terminal
 ├─ roles.py               # Definisi arketipe Council
 ├─ memory.py              # Episodic & semantic memory
 ├─ engine.py              # Debate engine (mode lama)
 ├─ personalities.py       # Agen untuk mode debate klasik
 ├─ storage.py             # Autosave JSON
 └─ clients.py             # Langfuse/Ollama client
debates/                   # Arsip Markdown output
memory/council_memory.db   # SQLite episodic memory
```

---

## 9. Troubleshooting

| Masalah | Penyebab Umum | Solusi |
| --- | --- | --- |
| CLI berhenti di fase awal | Model Ollama belum di-pull atau service belum jalan. | `ollama serve` dan pastikan semua model tersedia. |
| Langfuse warning | Variabel `.env` belum di-set. | Isi `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`. |
| Markdown log kosong | Sesi dihentikan sebelum fase selesai. | Biarkan council menyelesaikan semua fase atau cek file `.md` terbaru. |
| Memori tidak terpakai | Embedding gagal dibuat. | Pastikan `granite-embedding:latest` ter-install; periksa koneksi Ollama. |

---

## 10. Ringkasan

Council of Consciousness menyediakan kerangka multi-agen yang:
- Tematik dan konsisten melalui signature voice tiap archetype.
- Mampu mengingat diskusi terdahulu dan menggunakannya sebagai konteks.
- Memberi transparansi penuh lewat log Markdown dan memori episodik.
- Mudah dikontrol via CLI (non-interaktif ataupun wizard interaktif).

Dokumentasi ini dapat dijadikan acuan untuk melanjutkan pengembangan, integrasi UI, atau adaptasi ke domain organisasi lain.


