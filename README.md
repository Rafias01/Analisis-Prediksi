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
  

![image](https://github.com/user-attachments/assets/65f3c2a3-4641-4801-acb0-433b048ad6b0)

Gambar tersebut menunjukkan hasil dari pengecekan data kosong (missing values) pada sebuah dataset. Hasil yang ditampilkan menunjukkan bahwa tidak ada nilai kosong (missing values) di seluruh kolom. Yang artinya seluruh baris pada semua kolom memiliki data lengkap.

![image](https://github.com/user-attachments/assets/8ebfaa77-e711-4425-abd9-1f6b6686257f)

Gambar tersebut menunjukkan bahwa tidak terdapat data yang duplikat dalam dataset yang sedang dianalisis. Hal ini mengindikasikan bahwa setiap baris data bersifat unik dan tidak ada pengulangan entri yang sama persis. 

![image](https://github.com/user-attachments/assets/aac42739-1b2e-4bc1-9dd4-e83ac5adcd40)

Kode yang ditampilkan pada gambar tersebut merupakan implementasi fungsi Python untuk mendeteksi outlier (nilai pencilan) dalam sebuah dataset menggunakan metode IQR (Interquartile Range). Fungsi utama dalam kode ini adalah detect_outliers_iqr(data) yang menerima sebuah DataFrame data sebagai parameter. Di dalam fungsi ini, pertama-tama dibuat dictionary kosong bernama outlier_info untuk menyimpan informasi kolom mana saja yang memiliki outlier beserta jumlahnya.


Fungsi kemudian memproses semua kolom numerik bertipe int64 dan float64 dalam dataset menggunakan perulangan for. Untuk setiap kolom, dihitung kuartil pertama (Q1) dan kuartil ketiga (Q3), lalu dihitung rentang antar kuartil (IQR = Q3 - Q1). Berdasarkan nilai IQR, ditentukan batas bawah (lower_bound = Q1 - 1.5 * IQR) dan batas atas (upper_bound = Q3 + 1.5 * IQR). Nilai yang berada di luar rentang ini dianggap sebagai outlier. Kemudian, kode menghitung jumlah baris dalam kolom yang merupakan outlier dan jika ada, menyimpannya ke dalam dictionary outlier_info.

Setelah fungsi detect_outliers_iqr(df) dipanggil (dengan df sebagai DataFrame yang ingin dianalisis), hasilnya disimpan dalam outlier_results. Bagian berikutnya dari kode bertugas menampilkan hasil deteksi. Jika terdapat hasil (yakni dictionary outlier_results tidak kosong), maka akan dicetak daftar kolom beserta jumlah outlier-nya. Jika tidak ditemukan outlier, maka dicetak pesan bahwa tidak ada outlier dalam dataset.

Output pada gambar menunjukkan bahwa hanya ada satu kolom yang mengandung outlier, yaitu kolom Smoking dengan jumlah 904 outlier. Ini menunjukkan bahwa nilai-nilai dalam kolom tersebut memiliki distribusi yang ekstrem atau tidak normal, sehingga perlu perhatian lebih lanjut, misalnya dengan membersihkan data, mengkategorikan ulang, atau menggunakan metode statistik yang lebih tahan terhadap outlier.

![image](https://github.com/user-attachments/assets/30f7fb77-9591-4c96-8325-c72000cb1fa8)

Kode yang ditampilkan pada gambar bertujuan untuk mengubah representasi data biner dalam bentuk angka (0 dan 1) menjadi bentuk teks kategorikal yang lebih mudah dibaca, yaitu 'Tidak' dan 'Ya'. Langkah pertama adalah membersihkan nama kolom dengan df.columns.str.strip() untuk menghapus spasi di awal atau akhir nama kolom, yang sering kali terjadi akibat kesalahan saat membaca data dari file CSV atau Excel. Ini penting agar pemrosesan data selanjutnya tidak gagal karena nama kolom yang tidak konsisten.

Setelah itu, kode mendeteksi kolom-kolom dalam DataFrame df yang hanya berisi nilai 0 dan 1, yang merupakan ciri khas kolom biner. Hal ini dilakukan dengan list comprehension: kolom_biner = [col for col in df.columns if sorted(df[col].dropna().unique()) == [0, 1]]. Di sini, dropna() digunakan agar nilai NaN tidak ikut memengaruhi pengecekan, dan unique() diurutkan agar bisa dibandingkan langsung dengan daftar [0, 1].

Selanjutnya, kolom-kolom tersebut dicetak untuk memberi informasi kepada pengguna tentang kolom mana yang akan diubah. Kemudian, fungsi replace digunakan untuk mengganti nilai 1 menjadi 'Ya' dan 0 menjadi 'Tidak'. Proses ini membuat dataset lebih mudah dipahami, terutama saat digunakan untuk pelaporan atau visualisasi, karena 'Ya' dan 'Tidak' lebih intuitif dibandingkan angka biner. Tapi nanti nilainya akan diubah ke semula lagi pada proses encoding.

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
2. Distribusi **Cholesterol** terlihat agak menyebar rata namun sedikit condong ke kanan, mengisyaratkan adanya kecenderungan sebagian individu memiliki kadar kolesterol lebih tinggi dari rata-rata.
3. Pola distribusi pada **Heart Rate** menunjukkan keragaman yang cukup seimbang, tanpa ada nilai tertentu yang mendominasi, menandakan variasi detak jantung yang luas antar individu.  
4. Untuk **Exercise Hours Per Week**, histogram menunjukkan distribusi yang cenderung rata, menandakan bahwa partisipan memiliki kebiasaan olahraga yang bervariasi dari sangat sedikit hingga cukup intens. 
5. Histogram **Stress Level** menggambarkan distribusi yang hampir datar, menunjukkan bahwa responden tersebar secara seimbang di seluruh tingkat stres dari yang paling rendah hingga tertinggi.
6. Distribusi **Sedentary Hours Per Day** menunjukkan variasi yang lebar dan cukup merata, mengindikasikan bahwa durasi aktivitas duduk atau tidak bergerak sangat beragam dalam populasi ini.
7. Pada histogram **BMI**, terlihat distribusi yang cukup simetris dengan sedikit penumpukan pada rentang 23–27, yang mengindikasikan banyak individu berada dalam kisaran berat badan ideal.
9. **Triglycerides** memperlihatkan distribusi data yang merata dari nilai terendah hingga tertinggi, menunjukkan keberagaman kadar trigliserida yang signifikan di antara individu.
10. Untuk **Physical Activity Days Per Week**, penyebarannya relatif seimbang dari 0 hingga 7 hari, meskipun ada kecenderungan lebih banyak individu yang beraktivitas fisik sekitar 3 hari per minggu.
11. **Sleep Hours Per Day** memperlihatkan sebaran yang stabil dan merata antara 4 hingga 10 jam per hari, mencerminkan variasi pola tidur yang umum ditemui dalam populasi.


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
![image](https://github.com/user-attachments/assets/73e34506-30ed-4edf-b6e1-79762350c8cf)

Gambar tersebut memperlihatkan proses pembersihan data yang duplikan dan juga selanjutnya terdapat proses pendeteksian outlier pada dataset menggunakan metode Interquartile Range (IQR). Proses ini dilakukan dengan mendefinisikan fungsi detect_outliers_iqr(data) yang secara khusus ditujukan untuk mengevaluasi setiap kolom numerik dalam DataFrame. Fungsi ini hanya bekerja pada kolom dengan tipe data int64 dan float64, yang biasanya merepresentasikan data kuantitatif seperti umur, kadar kolesterol, denyut jantung, dan sebagainya.

Di dalam fungsi, langkah pertama adalah menghitung nilai kuartil pertama (Q1) dan kuartil ketiga (Q3) dari setiap kolom numerik. Selisih antara Q3 dan Q1 disebut sebagai IQR (Interquartile Range), yang mewakili rentang nilai tengah dari data. Berdasarkan IQR ini, ditentukan batas bawah dan batas atas untuk nilai normal. Nilai-nilai yang berada di luar rentang tersebut—lebih kecil dari Q1 - 1.5 × IQR atau lebih besar dari Q3 + 1.5 × IQR—dianggap sebagai outlier atau pencilan.

Setelah menentukan batas tersebut, fungsi mencari data yang termasuk ke dalam kategori outlier dan menghitung jumlahnya pada setiap kolom. Jika ditemukan outlier, jumlahnya dicatat dalam sebuah dictionary outlier_info yang berisi nama kolom dan jumlah outlier pada kolom tersebut. Namun, dalam kasus yang ditampilkan pada gambar, hasil pemeriksaan menunjukkan bahwa tidak ada kolom yang mengandung outlier, yang ditandai dengan output: "Tidak ditemukan outlier di dataset.". Hal ini terjadi karena saya telah melakukan penghapusan beberapa kolom yang ada.

### Encoding Kategorikal
![image](https://github.com/user-attachments/assets/1d0ec4c7-46e3-4bbf-9aca-3ea4817001fd)
kode tersebut menunjukkan proses encoding data kategorikal bertipe biner, yaitu nilai 'Ya' dan 'Tidak', menjadi nilai numerik 1 dan 0. Pertama, dilakukan pengecekan terhadap nilai unik pada kolom 'Heart Attack Risk' untuk memastikan bahwa kolom tersebut hanya memiliki dua nilai kategorikal, yakni 'Ya' dan 'Tidak'. Setelah itu, dibuat sebuah daftar berisi nama-nama kolom lain yang juga memiliki tipe data serupa, yaitu kolom_ya_tidak, yang mencakup fitur seperti "Diabetes", "Smoking", "Obesity", dan lainnya. Selanjutnya, dilakukan proses encoding dengan cara menerapkan fungsi map() pada setiap kolom dalam daftar tersebut, di mana nilai 'Ya' dikonversi menjadi angka 1 dan 'Tidak' menjadi 0. Proses ini penting dalam tahap data preparation karena algoritma machine learning umumnya membutuhkan input dalam bentuk numerik, sehingga data kategorikal perlu diubah ke format yang dapat diproses oleh model.

### One Hot Encoding
![image](https://github.com/user-attachments/assets/76e79a1d-4d2f-4e04-94c3-6cb99cbb0dd8)

kode tersebut menunjukkan proses one-hot encoding terhadap dua fitur kategorikal, yaitu "Sex" dan "Diet". Proses ini dilakukan dengan menggunakan fungsi pd.get_dummies() dari pustaka pandas, yang mengubah nilai kategorikal menjadi representasi numerik biner (0 dan 1) dalam bentuk kolom baru. Parameter drop_first=True digunakan agar salah satu kategori dari setiap fitur tidak disertakan (dibuang) untuk menghindari masalah multikolinearitas dalam model linier, seperti regresi logistik. Hasil encoding disimpan ke dalam variabel encoded. Kemudian, data hasil encoding tersebut digabungkan kembali dengan dataframe awal df_cleaned menggunakan pd.concat() di sepanjang kolom (axis=1). Terakhir, kolom asli "Sex" dan "Diet" dihapus dari dataframe karena sudah direpresentasikan oleh kolom hasil encoding, dengan menggunakan fungsi drop() dan inplace=True agar perubahan langsung diterapkan pada dataframe. Proses ini penting agar data kategorikal non-biner dapat digunakan oleh model machine learning yang hanya menerima input numerik.


### Data Splitting
![image](https://github.com/user-attachments/assets/a56dd698-98b8-4df8-8cb5-2a136636b8a5)

- 80% digunakan sebagai data latih (training data) untuk membangun model

- 20% sisanya digunakan sebagai data uji (testing data) untuk mengevaluasi kinerja model.

### Penjelasan Penggunaan SMOTE
SMOTE (Synthetic Minority Over-sampling Technique) digunakan ketika kita menghadapi masalah ketidakseimbangan kelas (class imbalance) dalam dataset, yaitu ketika jumlah sampel pada salah satu kelas (biasanya kelas minoritas) jauh lebih sedikit dibandingkan kelas lainnya. Ketidakseimbangan ini dapat menyebabkan model machine learning menjadi bias terhadap kelas mayoritas, karena model cenderung "bermain aman" dengan selalu memprediksi kelas yang paling sering muncul, sehingga performa terhadap kelas minoritas menjadi sangat buruk.

Dengan menggunakan SMOTE, kita menghasilkan data sintetis baru untuk kelas minoritas dengan cara membuat titik-titik data baru di sepanjang garis yang menghubungkan titik-titik data minoritas yang ada. Teknik ini berbeda dengan metode oversampling tradisional yang hanya menggandakan data, karena SMOTE menciptakan variasi yang lebih alami dan membantu model belajar pola yang lebih representatif dari kelas minoritas.

## Modelling
### 1. Algoritma Random Forest Classifier

Random Forest Classifier adalah salah satu algoritma pembelajaran mesin yang termasuk dalam metode ensemble learning, yang bertujuan untuk meningkatkan akurasi dan kestabilan model dengan menggabungkan prediksi dari banyak model sederhana. Dalam hal ini, Random Forest terdiri dari sejumlah pohon keputusan yang dibangun secara acak. Saat melakukan klasifikasi, setiap pohon memberikan suara (voting) untuk kelas tertentu, dan hasil akhir ditentukan berdasarkan mayoritas suara dari semua pohon tersebut. Terdapat beberapa parameter yang digunakan yaitu : 

- **random_state=42** : Parameter ini digunakan untuk memastikan bahwa proses pembentukan model bersifat konsisten dan dapat direproduksi. Parametir Ini penting karena Random Forest melibatkan proses acak, seperti pemilihan data secara acak untuk masing-masing pohon dan pemilihan subset fitur saat melakukan pemisahan cabang. Dengan menetapkan nilai random state, pengguna bisa mendapatkan hasil yang sama setiap kali kode dijalankan.
- **class_weight='balanced'** : Parameter ini  digunakan untuk menangani masalah data yang tidak seimbang (imbalanced dataset), yaitu kondisi ketika satu kelas memiliki jumlah data yang jauh lebih sedikit dibandingkan kelas lainnya. Dengan menggunakan pengaturan ini, model secara otomatis menghitung bobot untuk setiap kelas berdasarkan proporsi datanya di dalam data pelatihan. Bobot ini kemudian digunakan selama pelatihan agar kesalahan pada kelas minoritas mendapat penalti yang lebih besar, sehingga model terdorong untuk lebih memperhatikan kelas tersebut.
  
### 2. Algoritma Decision Tree
Decision Tree adalah salah satu algoritma pembelajaran mesin yang paling sederhana dan mudah dipahami, digunakan untuk tugas klasifikasi maupun regresi. Secara konseptual, decision tree bekerja dengan cara memecah dataset menjadi subset-subset yang lebih kecil berdasarkan fitur-fitur tertentu, hingga mencapai kondisi di mana data dalam suatu cabang cukup “murni” atau homogen. Struktur pohon dimulai dari simpul akar (root node), kemudian bercabang melalui simpul-simpul internal yang merepresentasikan keputusan berdasarkan nilai suatu fitur, dan berakhir pada simpul daun (leaf nodes) yang menyatakan hasil prediksi akhir. Terdapat parameter yang digunakan yaitu :

- **random_state=42** : Parameter ini berfungsi untuk memastikan bahwa proses pelatihan model bersifat deterministik, sehingga hasil yang diperoleh akan selalu sama jika kode dijalankan ulang. Ini penting karena meskipun Decision Tree tampak deterministik, dalam praktiknya terdapat proses pemilihan split yang bisa bersifat acak ketika beberapa split memiliki nilai informasi yang setara.


## Evaluasi dan Pemilihan Model
### Confusion Matrix
Confusion matrix adalah sebuah tabel yang digunakan untuk mengevaluasi kinerja model klasifikasi dengan cara membandingkan hasil prediksi model dengan nilai aktual (label sebenarnya). Confusion matrix memberikan informasi terperinci mengenai bagaimana model mengklasifikasikan data ke dalam masing-masing kelas, baik yang benar maupun yang salah.
![image](https://github.com/user-attachments/assets/e85662b3-719b-4df7-86ee-a2351bac605d)

Komponen dalam Confusion Matrix : 

- True Positive (TP):
   - Jumlah data yang diklasifikasikan benar sebagai kelas positif.
- False Positive (FP):
   - Jumlah data yang salah diklasifikasikan sebagai kelas positif, padahal sebenarnya negatif.
- False Negative (FN):
   - Jumlah data yang salah diklasifikasikan sebagai negatif, padahal sebenarnya positif.
- True Negative (TN):
   - Jumlah data yang diklasifikasikan benar sebagai kelas negatif.

Dari confusion matrix, beberapa metrik penting dapat dihitung:

- Accuracy: Persentase keseluruhan prediksi yang benar.
- Precision: Seberapa tepat model saat memprediksi kelas positif (TP / (TP + FP)).
- Recall (Sensitivity): Seberapa baik model dalam menangkap semua data positif (TP / (TP + FN)).
- F1-Score: Rata-rata harmonis antara precision dan recall. Cocok digunakan jika data tidak seimbang antar kelas.

Manfaat penggunaan confusion matrix : 
- Mengetahui jenis kesalahan klasifikasi (false positive vs false negative)
- Memahami keseimbangan kinerja model pada masing-masing kelas
- Menilai kualitas prediksi secara lebih menyeluruh dibanding hanya melihat akurasi saja
- Menjadi dasar dalam perbaikan model, misalnya dengan penyesuaian threshold, teknik sampling (SMOTE), atau pemilihan algoritma baru.

### Algoritma Random Forest 
![image](https://github.com/user-attachments/assets/ee6344d6-3ac3-45f4-9e5f-b0eb1e2db69c)
- Pada kelas 'Ya' (Berisiko Serangan Jantung) :
  - Precision = 0.35
    - : Artinya dari seluruh prediksi yang diklasifikasikan sebagai "Ya", hanya 35% yang benar-benar berisiko. Hal ini menunjukkan tingkat false positive yang cukup tinggi, yaitu banyak individu yang sebenarnya tidak berisiko tetapi diprediksi berisiko oleh model.
  - Recall = 0.19 :
    - Dari 628 individu yang sebenarnya berisiko serangan jantung, hanya 19% yang berhasil dikenali oleh model. Ini mengindikasikan kelemahan model dalam mendeteksi kasus berisiko (false negative masih sangat tinggi).
  - F1-Score = 0.25 :
    - Nilai F1-Score yang rendah mencerminkan performa keseluruhan yang buruk untuk kelas "Ya", karena baik precision maupun recall sama-sama rendah. Ini sangat penting diperhatikan karena bisa berdampak serius jika kasus berisiko tidak terdeteksi.

- Pada Kelas "Tidak" (Tidak Berisiko Serangan Jantung)
  - Precision = 0.64 :
    - Dari seluruh prediksi "Tidak", sekitar 64% benar-benar tidak berisiko. Sisanya merupakan false negative, yakni individu berisiko yang gagal dikenali.
  - Recall = 0.80 :
    - Dari total 1.125 individu yang benar-benar tidak berisiko, sebanyak 80% berhasil dikenali oleh model. Ini menunjukkan performa model cukup baik dalam mengenali individu yang sehat atau tidak berisiko.
  - F1-Score = 0.71 :
    - F1-Score yang cukup tinggi ini menandakan keseimbangan yang baik antara precision dan recall pada kelas "Tidak". Model lebih andal dalam mengklasifikasikan individu yang sehat daripada yang berisiko.   

### Algoritma Decision Tree
![image](https://github.com/user-attachments/assets/1babae41-9bc2-4184-b27a-ee80e39a1c8b)

- Pada kelas 'Ya' (Berisiko Serangan Jantung) :
  - Precision = 0.37 :
    - Dari semua prediksi yang menyatakan seseorang berisiko ("Ya"), hanya 37% benar-benar berisiko. Ini menunjukkan bahwa sebagian besar prediksi berisiko adalah salah (false positive).
  - Recall = 0.41 :
    - Dari 628 individu yang sebenarnya berisiko serangan jantung, sebanyak 41% berhasil dikenali oleh model. Tetapi masih kurang optimal karena lebih dari separuh individu berisiko tidak dikenali (false negative).
  - F1-Score = 0.39 :
    - Nilai F1-Score ini menunjukkan bahwa kinerja model dalam mendeteksi individu berisiko cukup rendah, meskipun lebih baik dari sebelumnya. Hal ini mencerminkan bahwa model masih belum cukup andal dalam menangani kasus kelas minoritas "Ya".
      
- Pada Kelas "Tidak" (Tidak Berisiko Serangan Jantung)
  - Precision = 0.65 :
    - Artinya, dari seluruh prediksi yang menyatakan "Tidak" (tidak berisiko), sebanyak 65% benar-benar sesuai, sedangkan sisanya (35%) adalah orang yang sebenarnya berisiko tetapi diklasifikasikan sebagai tidak berisiko (false negative).
  - Recall = 0.60 :
    - Dari total 1.125 individu yang benar-benar tidak berisiko, hanya 60% yang berhasil dikenali dengan benar oleh model. Ini menunjukkan bahwa kemampuan model dalam mengenali kelas "Tidak" menurun dibanding model Random Forest sebelumnya.
  - F1-Score = 0.63 :
    - Meskipun performa recall menurun, F1-Score tetap cukup stabil karena precision-nya tidak terlalu rendah. Ini mencerminkan bahwa model memiliki keseimbangan yang sedang antara precision dan recall untuk kelas "Tidak".

### Perbandingan Akurasi
![image](https://github.com/user-attachments/assets/7b770eac-5607-4c0f-95c7-36fe093d0277)
   
   Berdasarkan hasil perbandingan akurasi antara model Random Forest dan Decision Tree yang telah diterapkan teknik SMOTE, dapat disimpulkan bahwa model Random Forest memiliki akurasi yang lebih tinggi sebesar 58,13% dibandingkan dengan Decision Tree yang hanya mencapai 53,57%. Hal ini menunjukkan bahwa Random Forest lebih baik dalam memprediksi secara keseluruhan tetapi hasil tidak bagus dikarenakan dataset tidak bagus. Secara umum, dalam skenario klasifikasi yang berkaitan dengan deteksi penyakit serius seperti serangan jantung, kemampuan model untuk mengidentifikasi kelas minoritas (recall) menjadi lebih penting daripada hanya mengejar akurasi. Oleh karena itu, meskipun akurasinya lebih rendah, Decision Tree lebih layak dipertimbangkan karena memberikan hasil yang lebih seimbang dan sensitif terhadap pasien yang benar-benar berisiko. 

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
