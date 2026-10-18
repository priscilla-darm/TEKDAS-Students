# Coffee Shop Sales — Praktikum Business Intelligence

## Isi
- `audit_prepare_coffee.py`: audit kualitas data dan membuat data turunan tanpa mengubah Excel sumber atau menghapus baris otomatis.
- `app.py`: dashboard Streamlit interaktif tingkat lanjut (Executive Business Intelligence) dengan 6 tab analitis: Ringkasan Eksekutif, Analisis & Komparasi Gerai, Hierarki Kategori & Produk (Treemap & Ranking), Pola Waktu & Operasional (Heatmap Hari x Jam & Sesi Toko), Eksplorasi Data & Ekspor CSV, serta Metodologi & Audit Kualitas Data.
- Dilengkapi filter komprehensif: gerai, preset periode (Semua, Q1, Q2, Juni), kategori, tipe produk dinamis, hari kerja/akhir pekan, dan rentang jam transaksi.
- `requirements.txt`: daftar pustaka dependensi yang dibutuhkan.
- `data_prepared/`: CSV siap analisis, ringkasan bulanan, laporan audit.
- `ringkasan_bulanan.csv`: salinan ringkasan bulanan untuk dikumpulkan.
- `tabel_definisi_indikator.csv`: definisi empat KPI.
- `insight.md`: ringkasan insight bisnis dan batas interpretasi data.
- `screenshots/`: bukti tangkapan layar dashboard interaktif dengan variasi filter.

## Jalankan (Python 3.10+)

### macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python audit_prepare_coffee.py
streamlit run app.py
```

### Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python audit_prepare_coffee.py
streamlit run app.py
```

Letakkan `Coffee Shop Sales.xlsx` di folder proyek yang sama dengan `audit_prepare_coffee.py`, atau gunakan `--input` untuk menentukan path.

## Catatan metodologi
Satu baris berisi satu produk dan `transaction_id` unik; jangan menyebut transaksi tercatat sebagai pelanggan unik atau keseluruhan struk tanpa bukti tambahan. Mata uang tidak dinyatakan sumber. Harga bervariasi dipertahankan dan tidak otomatis dianggap salah.
