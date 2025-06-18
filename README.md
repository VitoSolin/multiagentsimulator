# Simulasi Multi-Agen E-commerce Retargeting

Selamat datang di proyek simulasi retargeting e-commerce menggunakan **Agent-Based Modeling (ABM)**. Dokumen ini bertujuan untuk menjelaskan cara kerja simulasi, konsep-konsep kunci di baliknya, dan bagaimana cara mempresentasikannya secara efektif.

## 🎯 Tujuan Proyek

Proyek ini mendemonstrasikan bagaimana interaksi antara berbagai "agen" pemasaran digital—masing-masing dengan strategi dan anggaran yang berbeda—dapat menciptakan dinamika pasar yang kompleks. Tujuannya adalah untuk memahami bagaimana strategi individu dan kerja sama (aliansi) memengaruhi keberhasilan dalam sebuah lingkungan lelang iklan (ad auction) yang kompetitif.

## 🧠 Konsep Kunci: Agent-Based Modeling (ABM)

Sebelum masuk ke kode, penting untuk memahami fondasi dari proyek ini: **Agent-Based Modeling (ABM)**.

-   **Apa itu ABM?** ABM adalah pendekatan pemodelan di mana kita mensimulasikan sebuah sistem dengan cara memodelkan perilaku individu-individu otonom ("agen") di dalamnya. Kita tidak mendefinisikan perilaku sistem secara keseluruhan, melainkan hanya aturan-aturan yang diikuti oleh setiap agen.
-   **Perilaku Emergent**: Fenomena menarik dalam ABM adalah "perilaku emergent". Ini adalah pola atau perilaku kompleks yang muncul di tingkat sistem sebagai hasil dari interaksi sederhana antar agen—pola yang tidak secara eksplisit diprogram. Dalam simulasi kita, contohnya adalah:
    -   Perang harga (bidding war) antara agen agresif.
    -   Keseimbangan pasar di mana agen yang berbeda memenangkan segmen pelanggan yang berbeda.
    -   Bagaimana aliansi dapat mengalahkan agen individu yang lebih kuat.

Saat presentasi, tekankan bahwa kita tidak hanya membuat skrip, tetapi **membangun sebuah dunia virtual** untuk melihat bagaimana strategi-strategi pemasaran saling berinteraksi.

## 🎭 Aktor dalam Simulasi

Ada dua jenis "aktor" utama dalam dunia virtual kita:

### 1. Agen Pemasaran (Marketing Agents)

Ini adalah "pemain utama" kita. Ada empat jenis, masing-masing dengan "kepribadian" (strategi) unik:

-   👨‍aggressive **Aggressive**: Tujuannya adalah memenangkan lelang sebanyak mungkin. Ia menawar dengan berani dan tidak terlalu pemilih terhadap pelanggan.
-   👩‍⚖️ **Conservative**: Sangat hati-hati dan efisien. Ia hanya akan menawar tinggi jika probabilitas konversi sangat menjanjikan. Agen ini lebih mementingkan profitabilitas (ROAS) daripada jumlah kemenangan.
-   🎯 **Retargeting-Focused**: Agen spesialis. Misinya adalah menargetkan ulang pelanggan yang sudah menunjukkan minat tinggi (misalnya, pernah memasukkan barang ke keranjang). Ia cenderung pasif sampai target emasnya muncul.
-   �� **Adaptive**: Agen oportunis cerdas. Strateginya tidak terpaku pada satu segmen, tetapi fokus utamanya adalah belajar dari performa masa lalunya (`ROAS`) untuk secara dinamis menyesuaikan tingkat agresivitas penawarannya.

### 2. Pelanggan (Customers)

Pelanggan bukanlah entitas pasif. Mereka memiliki siklus hidup yang dinamis, yang disebut `CustomerStage`:

1.  **Awareness**: Baru tahu tentang produk.
2.  **Interest**: Mulai tertarik, sering berkunjung.
3.  **Consideration**: Minat tinggi, sudah **memasukkan barang ke keranjang belanja (`cart_value` > 0)**.
4.  **Purchase**: Sudah pernah membeli.
5.  **Retention**: Pelanggan setia yang membeli berulang kali.
6.  **Churn Risk**: Sudah lama tidak aktif dan berisiko meninggalkan produk.

Perilaku pelanggan (mengklik, membeli, atau mengisi keranjang) dipengaruhi oleh iklan yang mereka lihat, yang pada gilirannya akan mengubah `stage` mereka.

## 📡 Komunikasi Antar Agen: Kerja Sama & Spionase

Salah satu fitur paling canggih dalam simulasi ini adalah kemampuan agen untuk "berkomunikasi" satu sama lain. Komunikasi ini bersifat **tidak langsung (indirect)**, artinya mereka tidak mengirim pesan eksplisit, melainkan mengamati tindakan satu sama lain dan bertindak berdasarkan informasi tersebut. Ada dua mekanisme utama:

### 1. Mekanisme Aliansi (Kerja Sama Strategis)
-   **Siapa**: `Conservative` dan `Adaptive`.
-   **Bagaimana**: Sebelum lelang, kedua agen ini menghitung "skor kecocokan" mereka terhadap pelanggan. Mereka kemudian saling membandingkan skor. Jika salah satu agen memiliki skor yang jauh lebih tinggi, agen sekutunya akan secara strategis "mengalah" (tidak menawar secara serius).
-   **Tujuan**: Ini adalah kerja sama tim yang cerdas untuk menghindari perang harga internal. Daripada saling menaikkan harga, mereka memastikan agen yang paling berpotensi menang bisa mendapatkan lelang dengan harga serendah mungkin, menghemat anggaran aliansi.

### 2. Saluran Komunikasi Publik (Spionase Pasar)
-   **Siapa**: Semua agen mempublikasikan, tetapi `Adaptive` adalah pendengar utama di masa lalu.
-   **Bagaimana**: Ada sebuah "papan pengumuman" digital (`communication_channel`) di mana setiap agen memposting rata-rata penawarannya. Agen `Adaptive` dapat "membaca" papan ini untuk melakukan "spionase pasar"—memahami seberapa agresif harga pasar saat ini.
-   **Tujuan**: Membantu agen membuat keputusan yang lebih cerdas. Namun, dalam versi terbaru, strategi agen `Adaptive` lebih memprioritaskan performa internal (ROAS) untuk memastikan ia tidak menjadi terlalu pasif di awal.

## ⚙️ Bagaimana Simulasi Bekerja: Alur per Langkah

Setiap "langkah" (timestep) dalam simulasi, yang kini berjalan dengan nilai mata uang **Rupiah (IDR)**, mengikuti alur kerja yang terstruktur:

1.  **Peluang Muncul**: Sebuah "peluang tayang iklan" (impression) muncul, menampilkan seorang pelanggan dengan `stage` dan riwayat tertentu.

2.  **Evaluasi & Komunikasi Internal**: Setiap agen pemasaran mengevaluasi pelanggan. Pada tahap ini, mekanisme komunikasi (terutama aliansi strategis) ikut berperan dalam menentukan langkah selanjutnya.

3.  **Pengajuan Penawaran & Lelang**: Agen mengajukan penawaran final mereka. Simulasi menggunakan **Second-Price Auction**, mekanisme lelang yang umum di dunia nyata. Pemenang hanya membayar senilai tawaran tertinggi kedua, yang mendorong agen untuk menawar nilai sebenarnya yang mereka yakini.

4.  **Interaksi Pelanggan**:
    -   Agen pemenang menampilkan iklannya.
    -   Pelanggan mungkin **mengklik** atau mengabaikannya.
    -   Jika mengklik tapi **tidak langsung membeli**, ada kemungkinan ia akan **memasukkan barang ke keranjang belanja**. Ini penting, karena inilah yang mengubah pelanggan ke tahap `CONSIDERATION` dan menciptakan target untuk agen `Retargeting-Focused`.
    -   Jika pelanggan membeli, agen mendapatkan pendapatan, dan `stage` pelanggan diperbarui.

5.  **Pembelajaran & Umpan Balik**:
    -   Semua agen memperbarui metrik mereka (budget yang tersisa, total pengeluaran, total pendapatan).
    -   Agen `Adaptive` menggunakan data baru ini (terutama ROAS) untuk menyesuaikan formulasi penawarannya di masa depan.
    -   Proses ini diulang ribuan kali.

## 📂 Struktur Kode `demo_multi_agent.py`

-   `MultiAgentRetargetingEnv`: Kelas ini adalah **"Dunia"** atau lingkungan simulasi. Ia mengelola data pelanggan, menjalankan mekanisme lelang, dan menghitung hasil dari setiap interaksi.
-   `SimpleAgent`: Kelas ini adalah **"Otak"** dari setiap agen pemasaran. Di sinilah semua logika strategi (kapan harus menawar tinggi/rendah), aliansi, dan penentuan aksi diimplementasikan.
-   `MultiAgentDemo`: Kelas ini adalah **"Sutradara"** dari keseluruhan demo. Ia menginisialisasi lingkungan dan para agen, menjalankan simulasi dari episode ke episode, dan mencetak ringkasan hasil di akhir setiap episode.

## 🚀 Cara Menjalankan Simulasi

1.  **Instalasi Dependensi**: Pastikan Anda memiliki semua pustaka yang diperlukan. Buka terminal di direktori proyek dan jalankan:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Unduh File Data & Model**: Dua file penting, `avazu_dev_pro.parquet` (data pelanggan) dan `ctr_model_pro_two.txt` (model prediksi), tidak termasuk dalam Git karena ukurannya.
    -   **Unduh kedua file tersebut dari Google Drive melalui tautan ini:**
        -   [**Tautan Google Drive: Unduh Data & Model**](https://drive.google.com/drive/folders/1ZLSY7XI8Y-wc1i_eRV9z_CjqBGk82E5C?usp=sharing)
    -   Setelah diunduh, **letakkan kedua file tersebut di dalam folder `multiagent/`**, sejajar dengan file `demo_multi_agent.py`.

3.  **Jalankan Demo**: Untuk menjalankan simulasi, eksekusi perintah berikut dari direktori utama:
    ```bash
    python multiagent/demo_multi_agent.py
    ```
    Setelah simulasi selesai, beberapa jendela grafik akan muncul secara otomatis. Tutup setiap jendela grafik untuk mengakhiri program sepenuhnya.

## 📊 Visualisasi & Analisis Hasil

Setelah simulasi selesai dijalankan, program akan secara otomatis membuat dan menampilkan tiga buah grafik untuk membantu analisis:

1.  **Distribusi Kemenangan Lelang per Episode**: Grafik ini menunjukkan agen mana yang paling sering memenangkan lelang di setiap episode. Ini membantu kita melihat dominasi pasar dan bagaimana strategi bersaing dari waktu ke waktu.
2.  **Performa Finansial Global per Episode**: Grafik ini menampilkan total pendapatan dan pengeluaran dari semua agen, serta metrik `ROAS` (Return on Ad Spend) secara keseluruhan. Ini memberikan gambaran tentang kesehatan "pasar" secara umum.
3.  **Sisa Anggaran Agen di Akhir Episode**: Grafik ini melacak sisa anggaran dari setiap agen setelah setiap episode selesai. Dari sini, kita bisa melihat agen mana yang paling efisien dalam membelanjakan anggarannya.
