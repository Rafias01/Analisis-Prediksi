# Laporan Proyek Machine Learning - Rafi Ananda Subekti

## Domain Proyek
Serangan jantung, atau myocardial infarction (MI), adalah kondisi medis serius yang terjadi ketika aliran darah ke sebagian otot jantung terhenti secara tiba-tiba, biasanya akibat penyumbatan pada arteri koroner. Penyumbatan ini sering disebabkan oleh penumpukan plak aterosklerotik yang terdiri dari lemak, kolesterol, dan zat lainnya. Ketika plak ini pecah, dapat terbentuk gumpalan darah yang menghambat aliran darah, menyebabkan kerusakan atau kematian jaringan otot jantung 

Serangan jantung merupakan salah satu penyebab utama kematian dan kecacatan di seluruh dunia. Menurut data dari National Health and Nutrition Examination Survey (NHANES) oleh CDC, diperkirakan 16,5 juta orang Amerika berusia di atas 20 tahun menderita penyakit arteri koroner, dengan prevalensi lebih tinggi pada pria dibandingkan wanita di semua kelompok usia.

Pencegahan serangan jantung melibatkan pengendalian faktor risiko melalui perubahan gaya hidup sehat, seperti berhenti merokok, menjaga berat badan ideal, berolahraga secara teratur, dan mengonsumsi makanan sehat. Selain itu, pengelolaan kondisi medis seperti hipertensi, diabetes, dan dislipidemia sangat penting.

## Business Understanding

### Problem Statements
1.  Apakah jenis kelamin, merokok, obesitas, diet dan Riwayat Penyakit Keluarga dapat menimbulkan penyakit serangan jantung?
2. Apakah ada hubungan yang signifikan antara usia dan risiko serangan jantung?
3. Pada rentang usia berapa risiko serangan jantung paling tinggi muncul?
   
### Goals
1. Mengetahui pengaruh faktor jenis kelamin, kebiasaan merokok, obesitas, dan riwayat penyakit jantung keluarga terhadap risiko terjadinya serangan jantung.
2. Mengidentifikasi hubungan antara usia dan risiko serangan jantung.
3. Menentukan kelompok usia dengan tingkat risiko serangan jantung tertinggi.
   
### Solution Statements
1. Melakukan eksplorasi dan analisis data secara statistik dan visual untuk mengidentifikasi pola, distribusi, serta keterkaitan antar variabel seperti jenis kelamin, kebiasaan merokok, obesitas, riwayat penyakit jantung keluarga, dan usia.
2. Menerapkan teknik Machine Learning seperti Decision Tree dan Random Forest untuk membangun model prediksi risiko serangan jantung, serta mengevaluasi seberapa besar kontribusi masing-masing faktor risiko terhadap outcome.
3. Menggunakan teknik penyeimbangan data seperti SMOTE untuk menangani ketidakseimbangan kelas dan meningkatkan akurasi prediksi model dalam mengidentifikasi individu berisiko tinggi.

### Metodologi
Analisis ini menggunakan pendekatan kuantitatif dengan metode analisis data eksploratif dan prediktif berbasis Machine Learning untuk mengetahui faktor-faktor yang memengaruhi risiko serangan jantung.

### Metrik
Metrik yang digunakan untuk mengevaluasi seberapa baik model klasifikasi merupakan confusion matrix. confusion matrix merupakan suatu metode yang digunakan untuk melakukan perhitungan akurasi pada konsep data mining.

## Data Understanding
Dataset diambil dari Kaggle ([link dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset)). Dataset ini berisi 1200 entri dengan dua kolom utama:
- Pada dataset tersebut dapat dilihat bahwa terdapat 26 kolom
- Pada dataset tersebut terdapat 8763 data

### Exploratory Data Analysis (EDA) - Univariate Analysis
- **Distribusi Kelas**:
       - ![alt text](?raw=true)

## Data Preparation
### Data Cleaning
- **Text Cleaning**: Menghapus tanda baca, angka, dan kata-kata tidak relevan (stopwords) menggunakan NLTK.
- **Stemming**: Menggunakan PorterStemmer untuk mengubah kata ke bentuk dasar (misalnya, "running" menjadi "run").
- **TF-IDF Vectorization**: Mengubah teks gejala menjadi vektor numerik untuk diproses model.
- **Label Encoding**: Mengubah label penyakit menjadi format numerik.

### Data Splitting
- Data dibagi menjadi 80% data latih dan 20% data uji menggunakan `train_test_split`.

### Normalisasi
- Fitur numerik (panjang teks dan jumlah kata) dinormalisasi menggunakan `MinMaxScaler` untuk menyeragamkan skala.

![Pipeline](images/pipeline.png)

## Modeling
### 1. Algoritma Support Vector Machine (SVM)
- **Kelebihan**: Efektif untuk data berdimensi tinggi seperti TF-IDF, robust terhadap noise.
- **Kekurangan**: Sensitif terhadap pemilihan kernel dan risiko overfitting pada data kompleks.
- **Parameter**: Menggunakan kernel linear dengan tuning Optuna (C=1.0).
- **Akurasi**: 98.5% (terbaik setelah tuning, dipilih untuk menghindari overfitting dibandingkan skor 100%).

### 2. Algoritma Logistic Regression
- **Kelebihan**: Sederhana, interpretable, cocok untuk data linear.
- **Kekurangan**: Kurang efektif pada hubungan non-linear.
- **Parameter**: Tuning dengan Optuna untuk parameter C.
- **Akurasi**: 98.1%.

### 3. Algoritma Multinomial Naive Bayes
- **Kelebihan**: Cepat, cocok untuk data teks.
- **Kekurangan**: Mengasumsikan independensi fitur, kurang akurat pada gejala tumpang tindih.
- **Akurasi**: 93%.

### 4. Algoritma XGBoost
- **Kelebihan**: Kuat menangani data kompleks, mendukung regularisasi.
- **Kekurangan**: Rentan terhadap kesalahan silang antar kelas.
- **Akurasi**: 91%.

### Pemilihan Model Terbaik
SVM dengan tuning Optuna dipilih karena:
- Akurasi tertinggi (98.5%).
- Konsistensi tinggi di semua kelas.
- Minim kesalahan pada penyakit penting seperti diabetes dan psoriasis.

![Perbandingan Akurasi Model](images/perbandingan-akurasi-model.png)

## Evaluation
### Metrik Evaluasi
- **Confusion Matrix**:

  ![Confussion Matrix](images/confussion-matrix.png)

  menunjukkan klasifikasi sempurna pada 21 dari 24 penyakit.

  
- **Classification Report**:

  ![Classification Report](images/classification-report.png)

  menunjukkan precision, recall, dan F1-score rata-rata 98.32% untuk SVM.
- **Cross-Validation**: Akurasi rata-rata SVM sebesar 98.23% pada 5-fold cross-validation.

### Studi Kasus
- **Input Gejala**: "increased thirst and hunger, frequent urination, unexplained weight loss, blurred vision, slow healing wounds".
- **Prediksi**: Diabetes (sesuai label aktual).
- **Interpretasi**: Sistem berhasil mengenali pola gejala diabetes dengan akurat, menunjukkan potensi sebagai alat bantu diagnosis awal.

### Kelemahan
- Model seperti Naive Bayes dan XGBoost kesulitan membedakan penyakit dengan gejala mirip (misalnya, drug reaction vs. GERD).
- Hasil prediksi harus divalidasi oleh profesional medis.

## Kesimpulan
1. **Sistem Klasifikasi**: Sistem NLP berbasis SVM mampu memprediksi penyakit dengan akurasi 98.5% berdasarkan gejala teks.
2. **Algoritma Terbaik**: SVM dengan tuning Optuna memberikan keseimbangan terbaik antara akurasi dan generalisasi.
3. **Dampak Klinis**: Sistem ini sangat potensial untuk diagnosis awal di lingkungan dengan sumber daya terbatas, tetapi tidak menggantikan dokter.

## Referensi
1. Dessi, D., Helaoui, R., Kumar, V., Recupero, D. R., & Riboni, D. (2021). TF-IDF vs word embeddings for morbidity identification in clinical notes: An initial study. *arXiv*. https://arxiv.org/abs/2105.09632
2. Kalra, S., Li, L., & Tizhoosh, H. R. (2019). Automatic classification of pathology reports using TF-IDF features. *arXiv*. https://arxiv.org/abs/1903.07406
3. Lai, L.-H., Lin, Y.-L., Liu, Y.-H., Lai, J.-P., Yang, W.-C., Hou, H.-P., & Pai, P.-F. (2024). The use of machine learning models with Optuna in disease prediction. *Electronics, 13*(23), 4775. https://doi.org/10.3390/electronics13234775
4. Rudd, J. M. (2017). Application of support vector machine modeling and graph theory metrics for disease classification. *arXiv*. https://arxiv.org/abs/1708.00122
