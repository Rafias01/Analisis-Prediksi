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
Dibawah ini merupakan kolom-kolom yang ada di dataset tersebut :

| Nama Kolom               | Deskripsi                                                                                                                        |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Patient ID               | identitas unik dari masing-masing pasien yang mana masing-masing pasien memiliki Patient ID yang berbeda-beda                    |
| Age                      | Usia dari masing-masing pasien                                                                                                   |
| Sex                      | Jenis kelamin pasien. Masing-masing terdiri dari Male (Laki-laki) dan Female (Perempuan)                                         |
| Cholesterol              | Banyaknya lemak dalam darah dalam satuan mg/dl yang mana cholesterol itu mempengaruhi apakah seseorang berpotentsi terkena                                         serangan jantung atau tidak                                                                                                      |
| Blood Pressure           | Blood pressure adalah tekanan yang diberikan darah terhadap dinding arteri saat jantung memompa dan beristirahat.                |
| Heart Rate               | Heart rate adalah jumlah detak jantung per menit yang menunjukkan seberapa cepat jantung bekerja.|
| Diabetes                 | Mengungkapkan apakah pasien mengidap diabetes dengan keterangan 1 dan 0 (1 = Ya, 0 Tidak)                                        |
| Family History           | Menyatakan apakah pasien memiliki riwayat keluarga yang mengidap penyakit serangan jantung dengan keterangan 1 dan 0 (1 = Ya, 0 Tidak)                                                                                                                                                        |
| Smoking                  | Menyatakan apakah pasien merokok atau tidak dengan keterangan 1 dan 0 (1 = Ya, 0 Tidak)                                          |
| Obesity                  |  Menyatakan apakah pasien obesitas atau tidak dengan keterangan 1 dan 0 (1 = Ya, 0 Tidak). Obesitas berpengaruh pada kesehatan                                      jantung                                                                                                                         |
| Alcohol Consumption      | Menyatakan apakah pengguna mengonsumsi alkohol atau tidak ('yes' atau 'no'). Ini merupakan salah satu faktor seseorang terkena                                     serangan jantung                                                                                                                 |
| Exercise Hours Per Week  | Jumlah waktu yang dihabiskan untuk melakukan olahraga dalam seminggu dalam satuan jam                                            |
| Diet                     | Diet atau pola makan seseorang yang dilakukan oleh seseorang dengan kriteria Health, Average dan Unhealthy                       |
| Previous Heart Problems  | Menyatakan apakah seseorang pernah terkena penyakit jantung sebelumnya dengan keterangan 1 dan 0 (1 = Ya, 0 Tidak).|
| Medication Use           | Penggunaan obat secara rutin (0 = tidak, 1 = ya). Bisa menunjukkan bahwa individu sedang dalam pengobatan untuk kondisi tertentu                                   seperti serangan jantung atau yang lainnya                                                                                       |
| Stress Level             | Tingkat stres  dalam skala (1–10). Stres kronis meningkatkan tekanan darah dan risiko penyakit jantung.                          |
| Sedentary Hours Per Day  | Jumlah jam duduk atau tidak aktif per hari. Gaya hidup sedentari adalah faktor risiko bagi penyakit jantung.                     |
| Income                   | Pendapatan tahunan. Keterangan ini dipakai untuk analisis bisnis                                                                 |
| BMI                      | Body Mass Index, dihitung dari berat badan dan tinggi badan. BMI tinggi menandakan obesitas yang mana salah satu faktor serangan                                   jantung                                                                                                                          |
| Triglycerides            | Kadar trigliserida (lemak) dalam darah (mg/dL). Kadar tinggi dikaitkan dengan peningkatan risiko penyakit serangan jantung.      |
| Physical Activity Days Per Week | Jumlah hari dalam seminggu individu berolahraga atau aktif secara fisik. Lebih banyak hari aktif biasanya berarti risiko                                           kesehatan yang lebih rendah.                                                                                              |
| Sleep Hours Per Day      | Rata-rata jam tidur per hari. Kualitas dan durasi tidur memengaruhi tekanan darah, kadar hormon, dan kesehatan jantung secara                                      keseluruhan.                                                                                                                     |
| Country                  | Negara pasien tinggal ini diperlukan untuk identitas pasien                                                                      |
| Continent                | Benua asal pasien. Ini berguna untuk agregasi data dalam cakupan geografis lebih luas.                                           |
| Hemisphere               | Belahan bumi dimana pasien tinggal. Yang mana terdiri dari Northern Hemisphere dan Southern Hemisphere                           |
| Heart Attack Risk        | Label target/variabel dependen (0 = risiko rendah, 1 = risiko tinggi).                                                           |


![image](https://github.com/user-attachments/assets/c2bf01db-c0c3-4e8e-8f9e-e58b3cde90a1)


Selanjutnya saya menghapus kolom-kolom yang dianggap tidak relevan, tidak berguna, atau berisiko mengganggu proses analisis dan pemodelan, sehingga dataset menjadi lebih fokus dan efisien untuk digunakan pada tahap berikutnya seperti eksplorasi data atau pelatihan model.
  

### Exploratory Data Analysis (EDA) - Univariate Analysis
- **Categorical**:
       ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/Univariate_Categorical1.png?raw=true)
       ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/Univariate_Categorical2.png?raw=true)
       ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/Univariate_Categorical3.png?raw=true)

Gambar di atas dapat diinterpretasikan sebagai berikut : 
1. Pada countplot jumlah berdasarkan **Sex** menunjukkan bahwa jumlah narasumber pria lebih besar dibandingkan dengan wanita.
2. Pada countplot jumlah berdasarkan **Diabetes** menunjukkan bahwa mayoritas narasumber mengidap penyakit diabetes.
3. Pada countplot jumlah berdasarkan **Family History** menunjukkan bahwa jumlah narasumber yang memiliki keluarga pengidap penyakit serangan jantung lebih sedikit dibandingkan narasumber yang memiliki keluarga pengidap penyakit serangan jantung.
4. Pada countplot jumlah berdasarkan **Smoking** menunjukkan bahwa jumlah narasumber yang merokok lebih banyak dibandingkan dengan yang tidak merokok.
5. Pada countplot jumlah berdasarkan **Diet** menunjukkan bahwa jumlah narasumber yang melakukan diet sehat, normal dan tidak sehat relatif sama jumlahnya.
6. Pada countplot jumlah berdasarkan **Alcohol Consumption** menunjukkan bahwa jumlah narasumber yang mengkonsumsi alkohol lebih banyak dibandingkan dengan yang tidak mengkonsumsi alkohol.
7. Pada countplot jumlah berdasarkan **Previous Heart Problems** menunjukkan bahwa jumlah narasumber yang pernah terkena dengan penyakit jantung lebih sedikit dibandingkan yang belum pernah terkena penyakit jantung.
8. Pada countplot jumlah berdasarkan **Obesity** menunjukkan bahwa jumlah narasumber yang obesitas relatif sama jumlahnya dengan yang tidak obesitas.
9. Pada countplot jumlah berdasarkan **Medication Use** menunjukkan bahwa jumlah narasumber yang menggunakan obat relatif sama dibandingkan dengan yang tidak menggunakan obat.
10. Pada countplot jumlah berdasarkan **Heart Attack Risk** menunjukkan bahwa jumlah narasumber yang Beresiko terkena penyakit jantung lebih sedikit dibandingkan dengan yang tidak beresiko terkena penyakit jantung
    
- **Numerical**:
  
  ![image](https://github.com/user-attachments/assets/4a128b9b-9a2e-4a9a-92f0-a4347bf3851d)
  
  ![image](https://github.com/user-attachments/assets/45cc5fe1-cf27-42ae-bd6e-1de92db40923)
  
  ![image](https://github.com/user-attachments/assets/70bbf608-c910-4c7d-9e1f-2832e7e9a926)
  
  ![image](https://github.com/user-attachments/assets/4c149ee6-08ac-4d18-bc0e-f4ccf71094b8)

Gambar di atas dapat diinterpretasikan sebagai berikut.
1. Histogram **Age** memperlihatkan penyebaran usia yang cukup merata tanpa dominasi kelompok usia tertentu, mencerminkan bahwa data mencakup individu dari usia muda hingga lanjut usia secara proporsional.
2.Distribusi **Cholesterol** terlihat agak menyebar rata namun sedikit condong ke kanan, mengisyaratkan adanya kecenderungan sebagian individu memiliki kadar kolesterol lebih tinggi dari rata-rata.
3. Pola distribusi pada **Heart Rate** menunjukkan keragaman yang cukup seimbang, tanpa ada nilai tertentu yang mendominasi, menandakan variasi detak jantung yang luas antar individu.
4. Untuk **Exercise Hours Per Week**, histogram menunjukkan distribusi yang cenderung rata, menandakan bahwa partisipan memiliki kebiasaan olahraga yang bervariasi dari sangat sedikit hingga cukup intens.
5. Histogram **Stress Level** menggambarkan distribusi yang hampir datar, menunjukkan bahwa responden tersebar secara seimbang di seluruh tingkat stres dari yang paling rendah hingga tertinggi.
6.Distribusi **Sedentary Hours Per Day** menunjukkan variasi yang lebar dan cukup merata, mengindikasikan bahwa durasi aktivitas duduk atau tidak bergerak sangat beragam dalam populasi ini.
7. Pada histogram **BMI**, terlihat distribusi yang cukup simetris dengan sedikit penumpukan pada rentang 23–27, yang mengindikasikan banyak individu berada dalam kisaran berat badan ideal.
8. **Triglycerides** memperlihatkan distribusi data yang merata dari nilai terendah hingga tertinggi, menunjukkan keberagaman kadar trigliserida yang signifikan di antara individu.
9. Untuk **Physical Activity Days Per Week**, penyebarannya relatif seimbang dari 0 hingga 7 hari, meskipun ada kecenderungan lebih banyak individu yang beraktivitas fisik sekitar 3 hari per minggu.
10. **Sleep Hours Per Day** memperlihatkan sebaran yang stabil dan merata antara 4 hingga 10 jam per hari, mencerminkan variasi pola tidur yang umum ditemui dalam populasi.


### Exploratory Data Analysis (EDA) - Multivariate Analysis
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/multivariate_analysis1.png?raw=true)
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/multivariate_analysis2.png?raw=true)
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/multivariate_analysis3.png?raw=true)
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/multivariate_analysis4.png?raw=true)
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/multivariate_analysis5.png?raw=true)

**Insight:**
1. Proporsi laki-laki yang memiliki risiko serangan jantung tinggi terlihat lebih banyak dibanding perempuan yang mana ini mengindikasikan bahwa jenis kelamin laki-laki mungkin menjadi faktor yang berkontribusi terhadap peningkatan risiko serangan jantung.

2. Individu yang memiliki kebiasaan merokok cenderung memiliki frekuensi lebih tinggi pada kategori risiko tinggi serangan jantung dibandingkan dengan non-perokok.

3. Individu yang mengalami obesitas terlihat memiliki jumlah risiko yang lebih tinggi terkena serangan jantung dibandingkan dengan Individu yang tidak obesitas.

4. Mereka dengan pola makan tidak sehat (Unhealthy) memiliki proporsi yang lebih besar dalam kategori risiko tinggi dibandingkan mereka yang memiliki pola makan sehat.

5. Individu yang memiliki riwayat serangan jantung dalam keluarga cenderung atau memiliki risiko yang lebih tinggi terkena serangan jantung dibandingkan dengan yang tidak memiliki riwayat keluarga terkena penyakit serangan jantung

   
   ![alt text](https://github.com/Rafias01/Analisis-Prediksi/blob/main/image/scatterplot.png?raw=true)
   
**Insight:**
1. Pada ScatterPlot tersebut dapat dilihat bahwa pria usia 50 tahun ke atas dengan kolesterol tinggi lebih banyak mengalami risiko serangan jantung.
2. Risiko serangan jantung tidak terbatas pada usia tua atau kolesterol tinggi, serangan jantung bisa muncul di segala usia dan level kolesterol.
3. Faktor jenis kelamin mungkin memengaruhi risiko, dengan pria menunjukkan lebih banyak kasus terkena serangan jantung.
   

## Data Preparation
### Data Cleaning
![image](https://github.com/user-attachments/assets/192ec309-00e3-46e1-91aa-018a7ffbe128)

   Gambar tersebut menunjukkan hasil dari pengecekan data kosong (missing values) pada sebuah DataFrame. Hasil yang ditampilkan menunjukkan bahwa tidak ada nilai kosong (missing values) di seluruh kolom. Yang artinya seluruh baris pada semua kolom memiliki data lengkap.

![image](https://github.com/user-attachments/assets/73e34506-30ed-4edf-b6e1-79762350c8cf)

Gambar tersebut memperlihatkan proses pembersihan data yang duplikan dan juga selanjutnya terdapat proses pendeteksian outlier pada dataset menggunakan metode Interquartile Range (IQR). Proses ini dilakukan dengan mendefinisikan fungsi detect_outliers_iqr(data) yang secara khusus ditujukan untuk mengevaluasi setiap kolom numerik dalam DataFrame. Fungsi ini hanya bekerja pada kolom dengan tipe data int64 dan float64, yang biasanya merepresentasikan data kuantitatif seperti umur, kadar kolesterol, denyut jantung, dan sebagainya.

Di dalam fungsi, langkah pertama adalah menghitung nilai kuartil pertama (Q1) dan kuartil ketiga (Q3) dari setiap kolom numerik. Selisih antara Q3 dan Q1 disebut sebagai IQR (Interquartile Range), yang mewakili rentang nilai tengah dari data. Berdasarkan IQR ini, ditentukan batas bawah dan batas atas untuk nilai normal. Nilai-nilai yang berada di luar rentang tersebut—lebih kecil dari Q1 - 1.5 × IQR atau lebih besar dari Q3 + 1.5 × IQR—dianggap sebagai outlier atau pencilan.

Setelah menentukan batas tersebut, fungsi mencari data yang termasuk ke dalam kategori outlier dan menghitung jumlahnya pada setiap kolom. Jika ditemukan outlier, jumlahnya dicatat dalam sebuah dictionary outlier_info yang berisi nama kolom dan jumlah outlier pada kolom tersebut. Namun, dalam kasus yang ditampilkan pada gambar, hasil pemeriksaan menunjukkan bahwa tidak ada kolom yang mengandung outlier, yang ditandai dengan output: "Tidak ditemukan outlier di dataset."

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
