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
- **Categorical**:
       ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/Univariate_Categorical1.png?raw=true)
       ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/Univariate_Categorical2.png?raw=true)
       ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/Univariate_Categorical3.png?raw=true)
  
- **Numerical**:
     ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/Univariate_Numerical1.png?raw=true)
     ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/Univariate_Numerical2.png?raw=true)
     ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/Univariate_Numerical3.png?raw=true)

### Exploratory Data Analysis (EDA) - Multivariate Analysis
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/multivariate_analysis1.png?raw=true)
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/multivariate_analysis2.png?raw=true)
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/multivariate_analysis3.png?raw=true)
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/multivariate_analysis4.png?raw=true)
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/multivariate_analysis5.png?raw=true)

   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/scatterplot.png?raw=true)



## Data Preparation
### Data Cleaning
- df.isnull().sum() = tidak ada data null
- df_cleaned = df_group.drop_duplicates() = tidak ada data duplikat

  def detect_outliers_iqr(data):
       outlier_info = {}
          for column in data.select_dtypes(include=['int64', 'float64']).columns:
              Q1 = data[column].quantile(0.25)
              Q3 = data[column].quantile(0.75)
              IQR = Q3 - Q1

              lower_bound = Q1 - 1.5 * IQR
              upper_bound = Q3 + 1.5 * IQR

  
              outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
              outlier_count = outliers.shape[0]

              if outlier_count > 0:
                  outlier_info[column] = outlier_count

    return outlier_info
  
        = Tidak ada outlier


### Data Splitting
- 80% digunakan sebagai data latih (training data) untuk membangun model

- 20% sisanya digunakan sebagai data uji (testing data) untuk mengevaluasi kinerja model.


## Modeling
### 1. Algoritma Random Forest Classifier
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/RandomForestClassification.png?raw=true)


### 2. Algoritma Decision Tree
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/DecisionTree.png?raw=true)

   
   Insight : 
   
      - Dari modelling yang kita lakukan kita melakukan modelling Algoritma Random Forest Classifier dan Decision Tree
      - Pada Random Forest memiliki akurasi sebesar 0.64 dan Decision Tree memiliki akurasi sebesar 0.55, yang mana ini menjelaskan bahwa akurasi Random Forest lebih besar dibandingkan dengan Decision Tree

## Evaluasi dan Pemilihan Model
### Perbandingan Akurasi
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/PerbandinganAkurasi.png?raw=true)
   
   Berdasarkan hasil perbandingan akurasi antara model Random Forest dan Decision Tree yang telah diterapkan teknik SMOTE, dapat disimpulkan bahwa model Random Forest memiliki akurasi yang lebih tinggi sebesar      63,55% dibandingkan dengan Decision Tree yang hanya mencapai 54,71%. Hal ini menunjukkan bahwa Random Forest lebih baik dalam memprediksi secara keseluruhan tetapi hasil tidak bagus dikarenakan dataset tidak     bagus. 



## Menjawab Problem 
1. Apakah jenis kelamin, merokok, obesitas, diet dan Riwayat Penyakit Keluarga dapat menimbulkan penyakit serangan jantung?
   
      Berdasarkan Countplot yang ada di bagian EDA - Multivariate Analysis dapat kita lihat berdasarkan grafik distribusi, jenis kelamin menunjukkan adanya kecenderungan bahwa laki-laki memiliki risiko serangan jantung yang lebih tinggi dibandingkan perempuan. Hal ini terlihat dari jumlah kasus risiko tinggi yang lebih dominan pada kelompok laki-laki. Selanjutnya, kebiasaan merokok juga memperlihatkan pola yang jelas, di mana individu yang merokok memiliki proporsi risiko serangan jantung yang lebih tinggi daripada yang tidak merokok, mendukung pemahaman bahwa merokok merupakan faktor risiko utama penyakit jantung.

      Obesitas menjadi faktor signifikan lainnya. Individu yang tergolong obesitas menunjukkan jumlah kasus serangan jantung yang lebih banyak, menunjukkan bahwa kelebihan berat badan dapat berkontribusi pada peningkatan tekanan darah, kolesterol, dan risiko komplikasi kardiovaskular lainnya. Dari segi pola makan, orang dengan diet yang tidak sehat juga cenderung memiliki risiko serangan jantung lebih tinggi, yang menunjukkan pentingnya menjaga asupan makanan dalam menjaga kesehatan jantung. Terakhir, riwayat penyakit jantung dalam keluarga juga memberikan kontribusi signifikan, di mana individu yang memiliki anggota keluarga dengan riwayat serangan jantung memiliki kecenderungan lebih besar untuk mengalami hal yang sama. Ini mengindikasikan adanya pengaruh genetik atau keturunan terhadap risiko penyakit jantung.

3. Apakah ada hubungan yang signifikan antara usia dan risiko serangan jantung?
     Pada bagian Exploratory Data Analysis (EDA) pada bagian analisis multivariat menggunakan scatterplot dapat diamati bahwa risiko serangan jantung menunjukkan tren peningkatan seiring bertambahnya usia. Pola ini semakin nyata terlihat pada kelompok usia di atas 50 tahun, di mana titik-titik yang merepresentasikan individu dengan risiko tinggi semakin padat. Hal ini menunjukkan bahwa usia merupakan salah satu faktor penting yang berkontribusi terhadap peningkatan risiko kardiovaskular. Seiring bertambahnya usia, terjadi penurunan elastisitas pembuluh darah, peningkatan tekanan darah, serta akumulasi plak pada arteri yang dapat memicu serangan jantung. Oleh karena itu, deteksi dini dan intervensi gaya hidup sehat menjadi sangat penting, terutama bagi individu yang memasuki usia paruh baya dan lansia.
   
4. Pada rentang usia berapa risiko serangan jantung paling tinggi muncul?
      Dalam tahap Exploratory Data Analysis (EDA), khususnya pada analisis multivariat yang divisualisasikan melalui scatterplot, tampak bahwa individu dalam rentang usia 50 hingga 70 tahun mendominasi area    dengan titik-titik berwarna oranye, yang merepresentasikan risiko serangan jantung yang tinggi. Hal ini menunjukkan bahwa sebagian besar kasus serangan jantung dalam dataset terjadi pada kelompok usia tersebut. Kepadatan titik-titik pada usia 50–70 mencerminkan bahwa risiko kardiovaskular cenderung meningkat secara signifikan seiring bertambahnya usia.
   
## Referensi
1. American Society for Preventive Cardiology. (2022). Ten things to know about ten cardiovascular disease risk factors. American Journal of Preventive Cardiology, 2(2), 100026.
2. Iskandar, I., Hadi, A., & Alfridsyah, A. (2017). Faktor Risiko Terjadinya Penyakit Jantung Koroner pada Pasien Rumah Sakit Umum Meuraxa Banda Aceh. AcTion: Aceh Nutrition Journal, 2(1), 32.
3. Karyatin, K. (2019). Faktor-Faktor Yang Berhubungan Dengan Kejadian Penyakit Jantung Koroner. Jurnal Ilmiah Kesehatan, 11(1), 37-43.
