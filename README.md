# Submission 2: Heart Failure Prediction

Nama: Ivana Lita

Username dicoding: ivana27lita

| **Komponen**             | **Deskripsi**                                                                                                                                                                                                                                                                                    |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**              | [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)                                                                                                                                                                                           |
| **Masalah**              | Kegagalan jantung adalah kondisi medis serius di mana jantung tidak dapat memompa darah secara efisien ke seluruh tubuh. Meskipun bisa mempengaruhi siapa saja, risiko lebih tinggi pada individu di atas 65 tahun. Penyakit ini memiliki tingkat kematian yang tinggi dibandingkan penyakit lain.     |
| **Solusi machine learning** | Mengembangkan model machine learning yang dapat menyediakan sistem prediksi yang dapat mendeteksi kondisi gagal jantung lebih awal agar dapat membantu mengurangi angka kematian akibat gagal jantung.                                                                                              |
| **Metode pengolahan**    | Data dalam proyek ini terdiri dari fitur kategorikal dan numerikal. Data kategorikal diubah menjadi one-hot encoding dan data numerikal dinormalisasi untuk memastikan skala yang konsisten. Proses ini melibatkan kode dalam `transform.py` untuk preprocessing.                                        |
| **Arsitektur model**     | Model prediksi dikembangkan dengan arsitektur yang mencakup preprocessing data melalui `transform.py`, serta pelatihan model menggunakan `trainer.py` dan `tuner.py` untuk optimasi. Model ini mengintegrasikan komponen dari `components.py` dan `pipeline.py` untuk prediksi kegagalan jantung berdasarkan fitur input. |
| **Metrik evaluasi**      | Evaluasi model dilakukan menggunakan metrik seperti akurasi, presisi, recall, dan F1-score. Akurasi mengukur proporsi prediksi yang benar, sementara presisi, recall, dan F1-score memberikan gambaran tentang keseimbangan antara prediksi positif dan negatif.                                       |
| **Performa model**       | Kinerja model dievaluasi dengan metrik akurasi, presisi, recall, dan F1-score, yang menunjukkan tingkat akurasi tinggi dan keseimbangan antara presisi dan recall. Hal ini menandakan efektivitas model dalam mendeteksi risiko kegagalan jantung dan mengurangi kesalahan prediksi.                   |
| **Opsi Deployment**      | Model ini di-deploy menggunakan platform Railway, yang menyediakan layanan gratis untuk deployment proyek machine learning. Proses deployment melibatkan `components.py` dan `pipeline.py` untuk integrasi dan pengelolaan pipeline.                                                                 |
| **Web App**              | [Heart Failure Prediction w/ Railway](https://heart-failure-pred-production.up.railway.app/v1/models/heartpred-model/metadata)                                                                                                                                                                      |
| **Monitoring**           | Monitoring dilakukan menggunakan Prometheus untuk memantau performa sistem, termasuk jumlah permintaan (requests) dan statusnya. Prometheus memungkinkan pemantauan yang mendetail terhadap aktivitas sistem.                                                                                      |
