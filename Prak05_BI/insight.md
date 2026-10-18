# Insight Coffee Shop Sales

Periode sumber: 1 Januari–30 Juni 2023. Semua angka menggunakan satuan nilai pada dataset; mata uang tidak dinyatakan sumber.

1. **Penjualan naik pada Juni, tetapi laju pertumbuhannya melambat.** Penjualan naik dari 156,727.76 pada Mei menjadi 166,485.88 pada Juni (6.23%). Jumlah transaksi tercatat naik dari 33,527 menjadi 35,352. Pertumbuhan Juni lebih rendah daripada pertumbuhan Mei terhadap April (31.77%). Tindak lanjut: pecah perubahan Juni menurut gerai dan kategori. Data ini tidak menjelaskan penyebab kenaikan.

2. **Hell’s Kitchen memiliki penjualan kumulatif tertinggi pada periode ini.** Nilai penjualannya 236,511.17, diikuti Astoria (232,243.91) dan Lower Manhattan (230,057.25). Selisih antargerai relatif kecil dibanding total. Tindak lanjut: bandingkan tren bulanan dan komposisi kategori per gerai sebelum menyimpulkan faktor kinerja.

3. **Kategori terbesar menurut nilai penjualan adalah Coffee.** Kontribusinya sekitar 38.6% dari total penjualan. Tindak lanjut: cek apakah komposisi kategori serupa pada masing-masing gerai; kontribusi tinggi tidak otomatis berarti margin atau laba tertinggi.

4. **Aktivitas transaksi paling banyak tercatat pada pukul 10:00.** Terdapat 18,545 transaction_id unik terakumulasi pada jam tersebut. Tindak lanjut: bila pertanyaan bisnisnya tentang antrean, kumpulkan durasi layanan atau catatan antrean. Volume transaksi saja tidak membuktikan antrean panjang.

## Batas interpretasi
- Satu baris memuat satu produk dan satu transaction_id unik; istilah yang aman adalah **transaksi tercatat**, bukan pelanggan unik atau keseluruhan struk.
- Tidak tersedia identitas pelanggan, biaya, stok, data promosi, atau durasi layanan. Karena itu loyalitas, laba, stockout, dampak promosi, dan antrean tidak dapat dipastikan dari file ini.
- Variasi harga produk tidak otomatis merupakan kesalahan; harga dibiarkan apa adanya dalam audit.
