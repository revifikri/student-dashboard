# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

Sebagai calon data scientist masa depan anda diminta untuk membantu Jaya Jaya Institut dalam menyelesaikan permasalahannya. Selain itu, mereka juga meminta Anda untuk membuatkan dashboard agar mereka mudah dalam memahami data dan memonitor performa siswa. 


## Permasalahan Bisnis
- Tingginya angka mahasiswa yang gagal menyelesaikan studi.
- Tidak adanya sistem prediktif untuk mengidentifikasi mahasiswa berisiko tinggi dropout.
- Kurangnya alat monitoring yang efektif untuk memantau performa akademik dan faktor risiko mahasiswa.
- Keterbatasan informasi visual untuk mendukung pengambilan keputusan strategis oleh pihak kampus.


## Cakupan Proyek
- Dataset: Student Performance Dataset dari Dicoding.
- Alur proyek end-to-end:
  - Eksplorasi dan pembersihan data.
  - Pra-pemrosesan dan transformasi fitur.
  - Penyeimbangan data (opsional).
  - Pelatihan model prediksi (Random Forest, Gradient Boosting, Logistic Regression).
  - Evaluasi model berdasarkan akurasi dan AUC.
  - Visualisasi dashboard interaktif menggunakan **Streamlit** dan **Tableau**.
  - Prediksi risiko dropout berdasarkan input karakteristik mahasiswa.
- Tools: Python, Pandas, Scikit-learn, Plotly, Streamlit, Tableau.

## Persiapan
- Sumber Data: https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv
### Setup Environment 
- Anaconda
conda create --name student-ds python=3.11
conda activate student-ds
pip install -r requirements.txt

- Terminal
mkdir student_dashboard
cd student_dashboard
pipenv install
pipenv shell
pip install -r requirements.txt

- Dashboard Tableau
Dashboard visualisasi dapat diakses melalui Tableau Public (online): https://public.tableau.com/shared/338CGPSX2?:display_count=n&:origin=viz_share_link


## Business Dashboard
### Link Dashboard
https://public.tableau.com/shared/338CGPSX2?:display_count=n&:origin=viz_share_link

### Penjelasan Dashboard:
1. KPI Panel
  High Risk Students: 677
  Dropout Rate: 0%
  Total Data: 4,424 mahasiswa
  Rata-rata Skor Akademik: 12.0

2. Distribusi Status Mahasiswa
  Dropout: 794
  Enrolled: 1421
  Graduate: 2209

3. Gender dan Status
  Gender mempengaruhi distribusi dropout.
  Mahasiswa laki-laki cenderung memiliki angka dropout lebih tinggi.

4. Age Group dan Status
  Mahasiswa usia 18–22 mendominasi status dropout.

5. Average Performance
  Dropout: Rata-rata nilai 10.68
  Enrolled: 11.94
  Graduate: 13.12

## Menjalankan Sistem Machine Learning
- Run Streamlit Terminal
cd student_dashboard
streamlit run student_dashboard.py

### Link Prototype
https://student-dashboard-g7xz2fkm2tf3ynsdu5boqy.streamlit.app/

## Kesimpulan
1. Faktor Risiko Dropout
  Skor akademik rendah.
  Evaluasi semester pertama yang buruk.
  Risiko ekonomi tinggi.
  Status perkawinan dan gender juga berperan.

2. Model Terbaik
  Model Gradient Boosting memberikan hasil akurasi terbaik dan AUC tinggi.
  Dilengkapi dengan fitur feature importance dan confusion matrix pada file ipynb.

3. Solusi 
  Sistem prediksi dropout berbasis ML terintegrasi dalam dashboard.
  Dapat digunakan oleh pihak akademik untuk melakukan intervensi awal.

4. Dampak Bisnis
  Deteksi awal terhadap mahasiswa berisiko tinggi dropout.
  Efisiensi dalam pengambilan keputusan akademik berbasis data.
  Potensi peningkatan retensi mahasiswa hingga >80%.

  ## Rekomendasi Action Items
  Berdasarkan hasil analisis data dan model prediksi yang telah dibangun dalam proyek ini, berikut adalah beberapa rekomendasi langkah konkret (action items) yang dapat dilakukan oleh institusi pendidikan untuk menurunkan angka dropout mahasiswa dan meningkatkan tingkat kelulusan:

---

### ✅ Action Item 1: Implementasi Sistem Peringatan Dini (Early Warning System)
Bangun sistem otomatis berbasis model machine learning untuk memberikan notifikasi kepada tim akademik apabila seorang mahasiswa masuk dalam kategori berisiko tinggi dropout.

- Frekuensi: mingguan atau bulanan
- Output: daftar mahasiswa berisiko tinggi lengkap dengan faktor penyebab utama
- Tools: integrasi dengan dashboard Streamlit/Tableau atau sistem informasi akademik (SIAKAD)

---

### ✅ Action Item 2: Intervensi Terarah untuk Mahasiswa Berisiko
Buat program intervensi khusus untuk mahasiswa dengan risiko tinggi dropout berdasarkan hasil prediksi model.

- Konseling akademik dan psikologis
- Bimbingan belajar atau remedial khusus
- Bantuan finansial (beasiswa, potongan UKT)

---

### ✅ Action Item 3: Monitoring Berkala melalui Dashboard Interaktif
Gunakan dashboard analitik (Streamlit/Tableau) untuk memantau tren dropout dan faktor-faktor penyebabnya secara real-time.

- Visualisasi dropout berdasarkan usia, gender, kinerja akademik
- Update data semesteran
- Akses oleh pimpinan fakultas dan bagian kemahasiswaan

---

### ✅ Action Item 4: Pelatihan untuk Dosen Wali & Tim Akademik
Berikan pelatihan kepada dosen wali dan pengelola akademik untuk memahami indikator risiko dropout dan cara membaca hasil prediksi dari dashboard.

- Materi: pemahaman fitur penting, interpretasi model, komunikasi ke mahasiswa
- Bentuk: webinar atau workshop internal

---

### ✅ Action Item 5: Evaluasi dan Pembaruan Model Secara Berkala
Untuk menjaga akurasi model seiring waktu, lakukan retraining model secara periodik menggunakan data terbaru.

- Frekuensi: setiap akhir semester atau tahun akademik
- Proses: preprocessing data baru, evaluasi ulang, deployment model baru
- Manfaat: model tetap relevan dengan tren terbaru

---

Dengan melaksanakan langkah-langkah ini secara konsisten, diharapkan institusi dapat secara signifikan menurunkan angka dropout dan meningkatkan efektivitas strategi retensi mahasiswa.
