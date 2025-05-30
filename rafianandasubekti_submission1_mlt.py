# -*- coding: utf-8 -*-
"""RafiAnandaSubekti_Submission1_MLT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1S9XNyuJ8HTtz4k4HpcTCOcU2gsB2ZqMx

# Predictive Analytic:  Obesity Levels

- **Nama:** Rafi Ananda Subekti
- **Email:** rafiasubekti@gmail.com
- **ID Dicoding:** MC009D5Y0612

## Kriteria Penilaian Tambahan - Domain Proyek
- Menjelaskan urgensi permasalahan dan pendekatan untuk menyelesaikannya.
- Menyertakan literatur atau studi sebagai pendukung.

## Kriteria Penilaian Tambahan - Business Understanding
- Menyediakan lebih dari satu strategi solusi, misalnya membandingkan beberapa model.

## Kriteria Penilaian - Data Understanding
- Melakukan eksplorasi untuk memahami karakteristik data, termasuk analisis statistik dan visualisasi :    
  - Diperlukan langkah-langkah seperti visualisasi distribusi fitur, korelasi antar variabel, serta analisis persebaran label target guna memahami struktur data secara menyeluruh.

## Kriteria Penilaian - Data Preparation
- Menjabarkan setiap langkah pengolahan data yang dilakukan sebelum pelatihan model

## Rubrik - Modeling
- Membandingkan keunggulan dan keterbatasan masing-masing algoritma
- Menggunakan beberapa model, model dengan performa terbaik dipilih dan dijelaskan alasannya

## Rubrik - Evaluation
- Menguraikan metrik evaluasi yang digunakan beserta fungsinya dalam mengukur kinerja model

# Domain Proyek
Serangan jantung, atau myocardial infarction (MI), adalah kondisi medis serius yang terjadi ketika aliran darah ke sebagian otot jantung terhenti secara tiba-tiba, biasanya akibat penyumbatan pada arteri koroner. Penyumbatan ini sering disebabkan oleh penumpukan plak aterosklerotik yang terdiri dari lemak, kolesterol, dan zat lainnya. Ketika plak ini pecah, dapat terbentuk gumpalan darah yang menghambat aliran darah, menyebabkan kerusakan atau kematian jaringan otot jantung

Serangan jantung merupakan salah satu penyebab utama kematian dan kecacatan di seluruh dunia. Menurut data dari National Health and Nutrition Examination Survey (NHANES) oleh CDC, diperkirakan 16,5 juta orang Amerika berusia di atas 20 tahun menderita penyakit arteri koroner, dengan prevalensi lebih tinggi pada pria dibandingkan wanita di semua kelompok usia.

Pencegahan serangan jantung melibatkan pengendalian faktor risiko melalui perubahan gaya hidup sehat, seperti berhenti merokok, menjaga berat badan ideal, berolahraga secara teratur, dan mengonsumsi makanan sehat. Selain itu, pengelolaan kondisi medis seperti hipertensi, diabetes, dan dislipidemia sangat penting.

# Business Understanding

## Problem Statements

Rumusan masalah dari masalah latar belakang diatas adalah:
1.  Apakah jenis kelamin, merokok, obesitas, diet dan Riwayat Penyakit Keluarga dapat menimbulkan penyakit serangan jantung?
2. Apakah ada hubungan yang signifikan antara usia dan risiko serangan jantung?
3. Pada rentang usia berapa risiko serangan jantung paling tinggi muncul?

## GOALS

Berdasarkan problem statements, berikut tujuan yang ingin dicapai pada proyek ini.
1. Mengetahui pengaruh faktor jenis kelamin, kebiasaan merokok, obesitas, dan riwayat penyakit jantung keluarga terhadap risiko terjadinya serangan jantung.
2. Mengidentifikasi hubungan antara usia dan risiko serangan jantung.
3. Menentukan kelompok usia dengan tingkat risiko serangan jantung tertinggi.

## Solution Statements

1. Melakukan eksplorasi dan analisis data secara statistik dan visual untuk mengidentifikasi pola, distribusi, serta keterkaitan antar variabel seperti jenis kelamin, kebiasaan merokok, obesitas, riwayat penyakit jantung keluarga, dan usia.
2. Menerapkan teknik Machine Learning seperti Decision Tree dan Random Forest untuk membangun model prediksi risiko serangan jantung, serta mengevaluasi seberapa besar kontribusi masing-masing faktor risiko terhadap outcome.
3. Menggunakan teknik penyeimbangan data seperti SMOTE untuk menangani ketidakseimbangan kelas dan meningkatkan akurasi prediksi model dalam mengidentifikasi individu berisiko tinggi.

## Metodologi

Analisis ini menggunakan pendekatan kuantitatif dengan metode analisis data eksploratif dan prediktif berbasis Machine Learning untuk mengetahui faktor-faktor yang memengaruhi risiko serangan jantung.

## Metrik

Metrik yang digunakan untuk mengevaluasi seberapa baik model klasifikasi merupakan confusion matrix. confusion matrix merupakan suatu metode yang digunakan untuk melakukan perhitungan akurasi pada konsep data mining.

# Data Understanding

Metrik yang digunakan untuk melihat hasil atau output dari model adalah Confussion Matrix. Confusion Matrix adalah sebuah tabel yang digunakan untuk mengevaluasi kinerja model klasifikasi. Salah satu menggunakan Confussion Matrix adalah Untuk melihat detail kesalahan prediksi.

## Mengimport Library

Data Understanding adalah tahap awal dalam proses analisis data yang bertujuan untuk memahami struktur, isi, kualitas, dan karakteristik data yang akan dianalisis.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

"""## Data Loading

Data Loading adalah proses mengimpor atau memuat data dari sumber eksternal ke dalam lingkungan analisis agar bisa diproses dan dianalisis lebih lanjut.
"""

#!/bin/bash
!curl -L -o heart-attack.zip\
  "https://www.kaggle.com/api/v1/datasets/download/iamsouravbanerjee/heart-attack-prediction-dataset"

!unzip heart-attack.zip -d heart-attack

df = pd.read_csv("/content/heart-attack/heart_attack_prediction_dataset.csv")
df

"""**Insight:**
- Pada tabel diatas dapat dilihat bahwa terdapat 26 kolom
- Pada tabel diatas juga terdapat 8763 data

### Deskripsi Variabel
"""

df.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Income', 'Blood Pressure'], inplace=True)

"""**Insight :**
  - Menghapus Kolom yang tidak akam kita pakai pada analisis   
"""

df.head()

df.info()

"""**Insight:**
- Terdapat 20 kolom setelah kita lakukan penghapusan pada kolom yang tidak diperlukan
- Terdapat 2 kolom dengan tipe data object dan 18 kolom dengan tipe float64 dan int64
"""

df.columns = df.columns.str.strip()

kolom_biner = [col for col in df.columns if sorted(df[col].dropna().unique()) == [0, 1]]

# Tampilkan kolom yang akan diubah (opsional)
print("Kolom dengan nilai 0/1 yang akan diubah:")
print(kolom_biner)

# Ubah 1 jadi 'Ya', 0 jadi 'Tidak'
df[kolom_biner] = df[kolom_biner].replace({1: 'Ya', 0: 'Tidak'})

# Tampilkan hasil
print(df.head())

df.shape

df_group = pd.DataFrame(df) #Ini kita lakukan agar yang tampil yang hanya tipe data numeric

"""### Deskripsi Statistik dari Data"""

df_group.describe()

"""Fungsi describe() digunakan untuk menampilkan ringkasan statistik dari setiap kolom dalam dataset, yang mencakup:

- Count menunjukkan jumlah data yang tersedia pada masing-masing kolom.
- Mean menyatakan nilai rata-rata dari data.
- Std adalah singkatan dari standar deviasi, yang menggambarkan sebaran atau variasi data
- Min menunjukkan nilai terkecil pada kolom tersebut.
- 25% adalah nilai kuartil pertama, yang berarti 25% data berada di bawah angka ini.
- 50% adalah kuartil kedua atau median, yaitu nilai tengah dari data.
- 75% merupakan kuartil ketiga, artinya 75% data berada di bawah nilai ini.
- Max menunjukkan nilai tertinggi dalam kolom tersebut.

**Insight:**
1. Umur (Age):

  * menunjukkan rentang usia dari 18 hingga 90 tahun.

  * Usia rata-rata nya adalah 53.7 tahun.


2. Cholesterol

  * Tingkat lemak dalam darah (mg/dL).

  * Nilai rata-rata cukup tinggi (259.88), mengindikasikan banyak individu memiliki kolesterol di atas ambang normal (sekitar 200 mg/dL).

3. Heart Rate :

  * Detak jantung dalam satuan denyut per menit (bpm).

  * Rata-rata detak jantung adalah 75 bpm, berada dalam kisaran normal (60–100 bpm).

4. Exercise Hours Per Week :

  * Jumlah jam olahraga yang dilakukan dalam seminggu.
  * Rata-rata malakukan selama 10 jam, tetapi ada individu dengan aktivitas hampir nol.

5. Stress Level :

  * Tingkat stres pada skala 1 (sangat rendah) hingga 10 (sangat tinggi).

  * Rata-rata berada di sekitar 5.47, artinya sebagian besar individu mengalami stres sedang.

6. Sedentary Hours Per Day :

  * Jumlah jam seseorang dalam posisi tidak aktif (misalnya duduk) per hari.

  * Rata-rata sekitar 6 jam, menunjukkan gaya hidup yang relatif tidak aktif

7. BMI (Body Mass Index) :

  * Indeks massa tubuh, digunakan untuk mengklasifikasikan berat badan (normal, overweight, obesitas)

  * Dalam tabel tersebut rata-rata BMI nya adalah 28.89, termasuk kategori overweight.

8. Triglycerides :

  * Jumlah trigliserida dalam darah (mg/dL).
  * Rata-rata: 417.68, jauh di atas ambang normal (kurang dari 150 mg/dL), menandakan risiko penyakit jantung tinggi.

9. Physical Activity Days Per Week :

  * Jumlah hari dalam seminggu seseorang melakukan aktivitas fisik.

  * Rata-rata 3.49 hari, menunjukkan tingkat aktivitas fisik yang masih tergolong rendah

10. Sleep Hours Per Day :

  * Durasi tidur per hari (dalam jam)

  * Memiliki tidur rata-rata 7 jam, sesuai dengan anjuran tidur sehat (7–9 jam per hari).

## Exploratory Data Analysis (EDA) - Univariate Analysis
"""

numerical_feature = ['Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week', 'Stress Level', 'Sedentary Hours Per Day', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week', 'Sleep Hours Per Day']
categorical_feature = ['Sex','Diabetes,' 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Diet', 'Previous Heart Problems', 'Medication Use', 'Heart Attack Risk ']

"""### Categorical Features"""

categorical_feature = [
    'Sex', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
    'Alcohol Consumption', 'Diet', 'Previous Heart Problems',
    'Medication Use', 'Heart Attack Risk'
]

deskripsi_kolom_kategorikal = [
    "Plot Jumlah dari Sex (Male/Female)",
    "Plot Jumlah dari Diabetes (Ya/Tidak)",
    "Plot Jumlah dari Family History with Heart Problems (Ya/Tidak)",
    "Plot Jumlah dari Smoking (Ya/Tidak)",
    "Plot Jumlah dari Obesity (Ya/Tidak)",
    "Plot Jumlah dari Alcohol Consumption (Ya/Tidak)",
    "Plot Jumlah dari Diet (Healthy/Average/Unhealthy)",
    "Plot Jumlah dari Previous Heart Problems (Ya/Tidak)",
    "Plot Jumlah dari Medication Use (Ya/Tidak)",
    "Plot Jumlah dari Heart Attack Risk (Ya/Tidak)"
]

# Buat subplot 5 baris x 2 kolom (10 plot)
fig, axes = plt.subplots(5, 2, figsize=(15, 20))
axes = axes.flatten()

# Loop dan buat plot
for i, kolom in enumerate(categorical_feature):
    sns.countplot(x=kolom, hue=kolom, data=df_group, ax=axes[i], palette='Set2', legend=False)
    axes[i].set_title(deskripsi_kolom_kategorikal[i], fontsize=10)
    axes[i].tick_params(axis="x", labelrotation=45)
    axes[i].tick_params(axis="both", which="major", labelsize=10)
    axes[i].set_xlabel("")

# Matikan sisa axes jika ada
for j in range(len(categorical_feature), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

"""Gambar di atas dapat diinterpretasikan sebagai berikut.
1. Pada countplot jumlah berdasarkan **Sex** menunjukkan bahwa jumlah narasumber pria lebih besar dibandingkan dengan wanita.
2. Pada countplot jumlah berdasarkan **Diabetes** menunjukkan bahwa mayoritas narasumber mengidap penyakit diabetes.
3. Pada countplot jumlah berdasarkan **Family History** menunjukkan bahwa jumlah narasumber yang memiliki keluarga pengidap penyakit serangan jantung lebih sedikit dibandingkan narasumber yang memiliki keluarga pengidap penyakit serangan jantung.
4. Pada countplot jumlah berdasarkan **Smoking** menunjukkan bahwa jumlah narasumber yang merokok lebih banyak dibandingkan dengan yang tidak merokok.
5. Pada countplot jumlah berdasarkan **Diet** menunjukkan bahwa jumlah narasumber yang melakukan diet sehat, normal dan tidak sehat relatif sama jumlahnya.
6. Pada countplot jumlah berdasarkan **Alcohol Consumption** menunjukkan bahwa jumlah narasumber yang mengkonsumsi alkohol lebih banyak dibandingkan dengan yang tidak mengkonsumsi alkohol.
7. Pada countplot jumlah berdasarkan **Previous Heart Problems** menunjukkan bahwa jumlah narasumber yang pernah terkena dengan penyakit jantung lebih sedikit dibandingkan yang belum pernah terkena penyakit jantung.
8. Pada countplot jumlah berdasarkan **Obesity** menunjukkan bahwa jumlah narasumber yang obesitas relatif sama jumlahnya dengan yang tidak obesitas.
9. Pada countplot jumlah berdasarkan **Medication Use** menunjukkan bahwa jumlah narasumber yang menggunakan obat relatif sama dibandingkan dengan yang tidak menggunakan obat.
10. Pada countplot jumlah berdasarkan **Heart Attack Risk** menunjukkan bahwa jumlah narasumber yang Beresiko terkena penyakit jantung lebih sedikit dibandingkan dengan yang tidak beresiko terkena penyakit jantung.

### Numerical Features
"""

for kolom in numerical_feature:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[kolom], kde=True, color='teal')
    plt.title(f'Distribusi dari {kolom}')
    plt.xlabel(kolom)
    plt.ylabel('Frekuensi')
    plt.tight_layout()
    plt.show()

"""Gambar di atas dapat diinterpretasikan sebagai berikut.
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

## Exploratory Data Analysis - Multivariate Analysis

### Membandingkan Heart Attack Risk dengan Sex, Smoking, Obesity, Diet dan Family History
"""

plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Heart Attack Risk', data=df, palette='Set1')
plt.title("Distribusi Jenis Kelamin terhadap Risiko Serangan Jantung")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Smoking', hue='Heart Attack Risk', data=df, palette='Set1')
plt.title("Distribusi Kebiasaan Merokok terhadap Risiko Serangan Jantung")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Obesity', hue='Heart Attack Risk', data=df, palette='Set1')
plt.title("Distribusi Jenis Kelamin terhadap Risiko Serangan Jantung")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Diet', hue='Heart Attack Risk', data=df, palette='Set1')
plt.title("Distribusi Diet terhadap Risiko Serangan Jantung")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Family History', hue='Heart Attack Risk', data=df, palette='Set1')
plt.title("Distribusi Family History terhadap Risiko Serangan Jantung")
plt.show()

"""**Insight:**
1. Proporsi laki-laki yang memiliki risiko serangan jantung tinggi terlihat lebih banyak dibanding perempuan yang mana ini mengindikasikan bahwa jenis kelamin laki-laki mungkin menjadi faktor yang berkontribusi terhadap peningkatan risiko serangan jantung.

2. Individu yang memiliki kebiasaan merokok cenderung memiliki frekuensi lebih tinggi pada kategori risiko tinggi serangan jantung dibandingkan dengan non-perokok.

3. Individu yang mengalami obesitas terlihat memiliki jumlah risiko yang lebih tinggi terkena serangan jantung dibandingkan dengan Individu yang tidak obesitas.

4. Mereka dengan pola makan tidak sehat (Unhealthy) memiliki proporsi yang lebih besar dalam kategori risiko tinggi dibandingkan mereka yang memiliki pola makan sehat.

5. Individu yang memiliki riwayat serangan jantung dalam keluarga cenderung atau memiliki risiko yang lebih tinggi terkena serangan jantung dibandingkan dengan yang tidak memiliki riwayat keluarga terkena penyakit serangan jantung

### Hubungan Usia (Age) dan Kolesterol (Cholesterol) terhadap Risiko Serangan Jantung (Heart Attack Risk) yang dibedakan berdasarkan jenis kelamin (Sex)
"""

g = sns.FacetGrid(df, col="Sex", hue="Heart Attack Risk", height=6, aspect=1.2)
g.map_dataframe(sns.scatterplot, x="Age", y="Cholesterol")
g.add_legend()
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Usia vs Kolesterol Berdasarkan Risiko Serangan Jantung, berdasarkan Jenis Kelamin")
plt.show()

"""**Insight:**
1. Pada ScatterPlot tersebut dapat dilihat bahwa pria usia 50 tahun ke atas dengan kolesterol tinggi lebih banyak mengalami risiko serangan jantung.
2. Risiko serangan jantung tidak terbatas pada usia tua atau kolesterol tinggi, serangan jantung bisa muncul di segala usia dan level kolesterol.
3. Faktor jenis kelamin mungkin memengaruhi risiko, dengan pria menunjukkan lebih banyak kasus terkena serangan jantung.

# Data Preparation

### Data Cleaning

#### Menangani Data Duplikat
"""

df.isnull().sum()

df_cleaned = df_group.drop_duplicates()

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

# Deteksi outlier
outlier_results = detect_outliers_iqr(df)

# Tampilkan hasil
if outlier_results:
    print("Kolom yang mengandung outlier:")
    for col, count in outlier_results.items():
        print(f"- {col}: {count} outlier")
else:
    print("Tidak ditemukan outlier di dataset.")

"""## Encoding Kategorikal

Melakukan proses Encoding Kategorikal yang mana proses ini dilakukan terhadap beberapa variabel yang hanya berisi antara Ya dan Tidak, yaitu pada variable:
* **Diabetes** (Apakah mengidap penyakit diabetes?),
* `**Family History** (Apakah ada anggota keluarga yang mengidap penyakit serangan jantung?),
* **Smoking** (Apakah merokok?),
* **Obesity** (Apakah mengidap obesitas?)
* **Alcohol Consumption** (Apakah mengonsumsi alkohol?)
* **Previous Heart Problems** (Apakah pernah terkena penyakit jantung sebelumnya?)
* **Medication Use** (Apakah menggunakan obat?)
"""

df_cleaned['Heart Attack Risk'].unique()

kolom_ya_tidak = ["Diabetes", "Family History", "Smoking", "Obesity",
                  "Alcohol Consumption", "Previous Heart Problems",
                  "Medication Use"]

df_cleaned[kolom_ya_tidak] = df_cleaned[kolom_ya_tidak].apply(lambda col: col.map({"Ya": 1, "Tidak": 0}))

"""## One Hot Encoding

One Hot Encoding dilakukan terhadap 2 variabel, One Hot Encoding ini dilakukan terhadap categorical selain yang isinya adalah Ya atau Tidak

* Sex : Male, Female
* Diet : Healthy, Average, Unhealthy
"""

encoded = pd.get_dummies(df_cleaned[["Sex", "Diet"]], drop_first=True)

df = pd.concat([df_cleaned, encoded], axis=1)

df.drop(columns=["Sex", "Diet"], inplace=True)

# Menampilkan data teratas
df.head()

list(df.columns)

"""## Split train test

Karena variabel Heart Attack Risk merupakan target yang ingin diprediksi, langkah selanjutnya adalah menghapus kolom tersebut dari dataset dan menyimpannya ke dalam variabel khusus untuk proses prediksi
"""

X = df.drop('Heart Attack Risk', axis=1)  # Fitur
y = df['Heart Attack Risk']              # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Cek hasil
print(f"Data latih: {len(X_train)}, Data uji: {len(X_test)}")

"""Selanjutnya, data akan dibagi menjadi dua bagian, yaitu:

- 80% digunakan sebagai data latih (training data) untuk membangun model

- 20% sisanya digunakan sebagai data uji (testing data) untuk mengevaluasi kinerja model.

# Modeling
"""

from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model_performance(y_true, y_pred, title, class_labels=None):
    if class_labels is None:
        class_labels = ['Ya', 'Tidak']

    # Tampilkan laporan klasifikasi dengan nama kelas
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Visualisasi confusion matrix dengan label kategori
    fig, ax = plt.subplots(figsize=(10, 5))
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, labels=class_labels)

    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_yticklabels(class_labels)

    ax.grid(False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

"""##Model Dengan Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # pastikan imblearn sudah di-install
from sklearn.model_selection import train_test_split

# 1. Menerapkan SMOTE untuk oversampling kelas minoritas
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 2. Membuat model Random Forest dengan class_weight
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_balanced, y_train_balanced)

# 3. Prediksi dan evaluasi
y_pred_rf = rf_model.predict(X_test)

# Evaluasi: confusion matrix + classification report
evaluate_model_performance(y_test, y_pred_rf, title="Random Forest (SMOTE + Balanced Weight)")

print(classification_report(y_test, y_pred_rf))

"""## Model Development dengan Algoritma Decision Tree"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Terapkan SMOTE untuk menyeimbangkan data pelatihan
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Membuat model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

# Melatih model dengan data train hasil SMOTE
dt_model.fit(X_train_bal, y_train_bal)

# Prediksi pada data test
y_pred_dt = dt_model.predict(X_test)

# Evaluasi hasil
evaluate_model_performance(y_test, y_pred_dt, title="Decision Tree + SMOTE")

print("=== Decision Tree Results with SMOTE ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

"""Insight :
- Dari modelling yang kita lakukan kita melakukan modelling Algoritma Random Forest Classifier dan Decision Tree
- Pada Random Forest memiliki akurasi sebesar 0.64 dan Decision Tree memiliki akurasi sebesar 0.55, yang mana ini menjelaskan bahwa akurasi Random Forest lebih besar dibandingkan dengan Decision Tree

# Evaluasi dan Pemilihan Model
"""

acc_rf = accuracy_score(y_test, y_pred_rf)
acc_dt = accuracy_score(y_test, y_pred_dt)

# Buat dataframe dari hasil akurasi
results_df = pd.DataFrame({
    'Model': ['Random Forest', 'Decision Tree'],
    'Accuracy': [acc_rf, acc_dt]
})

# Urutkan dataframe berdasarkan akurasi (descending)
results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

print(results_df)

"""Dari kedua algoritma yang ada, algoritma yang paling bagus adalah algoritma Random Forest"""

model_names = ['Random Forest', 'Decision Tree']
accuracies = [acc_rf, acc_dt]

plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=accuracies, palette='viridis')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Perbandingan Akurasi Model (SMOTE)')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.4f}", ha='center', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""Berdasarkan hasil perbandingan akurasi antara model Random Forest dan Decision Tree yang telah diterapkan teknik SMOTE, dapat disimpulkan bahwa model Random Forest memiliki akurasi yang lebih tinggi sebesar 63,55% dibandingkan dengan Decision Tree yang hanya mencapai 54,71%. Hal ini menunjukkan bahwa Random Forest lebih baik dalam memprediksi secara keseluruhan tetapi hasil tidak bagus dikarenakan dataset tidak bagus.

# Menjawab Problems

### 1. Apakah jenis kelamin, merokok, obesitas, diet dan Riwayat Penyakit Keluarga dapat menimbulkan penyakit serangan jantung?

Untuk menjawab permasalahan ini, kita harus melakukan analisis pada kolom Sex, Smoking, Obesity, Diet, dan Family History berdasarkan Heart Attack Risk

Berdasarkan Countplot yang ada di bagian EDA - Multivariate Analysis dapat kita lihat berdasarkan grafik distribusi, jenis kelamin menunjukkan adanya kecenderungan bahwa laki-laki memiliki risiko serangan jantung yang lebih tinggi dibandingkan perempuan. Hal ini terlihat dari jumlah kasus risiko tinggi yang lebih dominan pada kelompok laki-laki. Selanjutnya, kebiasaan merokok juga memperlihatkan pola yang jelas, di mana individu yang merokok memiliki proporsi risiko serangan jantung yang lebih tinggi daripada yang tidak merokok, mendukung pemahaman bahwa merokok merupakan faktor risiko utama penyakit jantung.

Obesitas menjadi faktor signifikan lainnya. Individu yang tergolong obesitas menunjukkan jumlah kasus serangan jantung yang lebih banyak, menunjukkan bahwa kelebihan berat badan dapat berkontribusi pada peningkatan tekanan darah, kolesterol, dan risiko komplikasi kardiovaskular lainnya. Dari segi pola makan, orang dengan diet yang tidak sehat juga cenderung memiliki risiko serangan jantung lebih tinggi, yang menunjukkan pentingnya menjaga asupan makanan dalam menjaga kesehatan jantung. Terakhir, riwayat penyakit jantung dalam keluarga juga memberikan kontribusi signifikan, di mana individu yang memiliki anggota keluarga dengan riwayat serangan jantung memiliki kecenderungan lebih besar untuk mengalami hal yang sama. Ini mengindikasikan adanya pengaruh genetik atau keturunan terhadap risiko penyakit jantung.

### 2. Apakah ada hubungan yang signifikan antara usia dan risiko serangan jantung?

Pada bagian Exploratory Data Analysis (EDA) pada bagian analisis multivariat menggunakan scatterplot dapat diamati bahwa risiko serangan jantung menunjukkan tren peningkatan seiring bertambahnya usia. Pola ini semakin nyata terlihat pada kelompok usia di atas 50 tahun, di mana titik-titik yang merepresentasikan individu dengan risiko tinggi semakin padat. Hal ini menunjukkan bahwa usia merupakan salah satu faktor penting yang berkontribusi terhadap peningkatan risiko kardiovaskular. Seiring bertambahnya usia, terjadi penurunan elastisitas pembuluh darah, peningkatan tekanan darah, serta akumulasi plak pada arteri yang dapat memicu serangan jantung. Oleh karena itu, deteksi dini dan intervensi gaya hidup sehat menjadi sangat penting, terutama bagi individu yang memasuki usia paruh baya dan lansia.

### 3. Pada rentang usia berapa risiko serangan jantung paling tinggi muncul?

Dalam tahap Exploratory Data Analysis (EDA), khususnya pada analisis multivariat yang divisualisasikan melalui scatterplot, tampak bahwa individu dalam rentang usia 50 hingga 70 tahun mendominasi area dengan titik-titik berwarna oranye, yang merepresentasikan risiko serangan jantung yang tinggi. Hal ini menunjukkan bahwa sebagian besar kasus serangan jantung dalam dataset terjadi pada kelompok usia tersebut. Kepadatan titik-titik pada usia 50–70 mencerminkan bahwa risiko kardiovaskular cenderung meningkat secara signifikan seiring bertambahnya usia.

# Referensi

1. American Society for Preventive Cardiology. (2022). Ten things to know about ten cardiovascular disease risk factors. American Journal of Preventive Cardiology, 2(2), 100026.
2. Iskandar, I., Hadi, A., & Alfridsyah, A. (2017). Faktor Risiko Terjadinya Penyakit Jantung Koroner pada Pasien Rumah Sakit Umum Meuraxa Banda Aceh. AcTion: Aceh Nutrition Journal, 2(1), 32.
3. Karyatin, K. (2019). Faktor-Faktor Yang Berhubungan Dengan Kejadian Penyakit Jantung Koroner. Jurnal Ilmiah Kesehatan, 11(1), 37-43.
"""