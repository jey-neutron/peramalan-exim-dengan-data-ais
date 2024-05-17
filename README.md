# Tentang
Penelitian ini akan menggunakan data AIS sebagai sumber data baru. Data AIS yang digunakan merupakan data AIS yang dikirim kapal di wilayah Indonesia selama periode dua tahun (2019-2020). Kemudian data tersebut dilakukan preprocessing untuk menghilangkan noise dan mendapatkan data yang terkait dengan aktivitas ekspor impor. Untuk dapat meramalkan statistik ekspor impor bulanan, data AIS diagregasi menjadi indikator bulanan terkait ekspor impor. Beberapa indikator tersebut akan diseleksi sebagai variabel prediktor untuk meramalkan statistik ekspor dan impor Indonesia.
Metode peramalan yang digunakan adalah metode Artificial Neural Network (ANN). Tuning parameter akan dilakukan untuk mendapatkan parameter model ANN yang optimal menggunakan random search. Hasil dengan model peramalan tersebut akan dibandingkan dengan metode peramalan tradisional menggunakan ARIMA. Error yang dihasilkan kedua model akan dihitung menggunakan RMSE dan MAPE. Model yang memiliki error terkecil akan dipilih sebagai model yang terbaik untuk meramalkan atau memprediksi nilai dan volume ekspor impor Indonesia.

Dalam folder [`Sourcecode+data`](Soucecode+data/) terdapat data dan source code dalam bentuk file jupyter notebook yang digunakan dalam penyusunan skripsi "Pemanfaatan Data AIS dalam Pemodelan Nowcasting Statistik Ekspor Impor Indonesia".

# Tahap pengumpulan data
Data statistik ekspor impor dikumpulkan dari web BPS ([https://www.bps.go.id](https://www.bps.go.id)), sedangkan data AIS dikumpulkan dari database Geomesa dari penyedia data AIS exactEarth yang disediakan oleh UN Global Platform. Database tersebut dapat diakses melalui Jupyter Hub ([https://location.officialstatistics.org](https://location.officialstatistics.org/hub/login)) yang memerlukan akun UNGP [^1]. 

[^1]: Sejak bulan Juni 2021, akses ke database AIS diubah melalui Jupyter Hub ([https://notebooks.officialstatistics.org](https://notebooks.officialstatistics.org)) dan GitLab ([https://code.officialstatistics.org](https://code.officialstatistics.org)).


### Data AIS
File [`_1_tradeIndicatorsClass_Indonesia.ipynb`](Sourcecode+data/_1_tradeIndicatorsClass_Indonesia.ipynb) merupakan source code untuk mengakses data AIS dan perlu dihubungkan dengan kernel PySpark Jupyter Hub di UNGP. Data AIS yang tersedia berukuran besar sehingga data tersebut langsung diolah disana. File ini juga akan membentuk indikator terkait ekspor impor dari data AIS, dengan sebelumnya melakukan preprocessing/filtering pada data AIS. Tahap filtering yang dilakukan yaitu:
- Memfilter pesan AIS yang dikirim kapal di wilayah bounding box Indonesia pada rentang waktu tertentu
    ```python
      # rentang waktu
      df = df.filter((F.col("dtg") > F.unix_timestamp(F.lit(startDate)).cast('timestamp')) & 
                     (F.col("dtg") < F.unix_timestamp(F.lit(endDate)).cast('timestamp')) )
      # filter bounding box
      filterExpr = "st_contains(st_makeBBox({0},{1},{2},{3}),position)".format(areaLongitude['lLim'], 
                                areaLatitude['lLim'], areaLongitude['uLim'], areaLatitude['uLim'])
      df = df.filter(F.expr(filterExpr))
    ```
- Filter 1: MMSI kapal yang valid
    ```python
      df = df.filter((df["mmsi"]>=100000000) & (df["mmsi"]<=999999999) )
    ```
- Filter 2: kapal yang melakukan pelayaran
    ```python
      df = aisClass.movingShips(df = df, latDiff = 0.1, longDiff = 0.1)
    ```
- Filter 3: status kapal berlabuh
    ```python
      df = df.filter(df.nav_status.rlike('Moored|Anchor|Manoeuvrability'))\
    ```
- Filter 4: non-zero draught
    ```python
      df = df.filter(df.draught > 0)
    ```
- Filter 5: tipe kapal yang relevan
    ```python
      df = df.filter(df.vessel_type.rlike('Cargo|Tanker'))
    ```
- Filter 6: kapal yang berada di Pelabuhan
    ```python
      df = aisClass.definePortNo(df = df, portCoords = portCoords)
    ```

Indikator yang dibentuk yaitu:
- Time in port (`timeInPort()`)               : menghitung lama waktu kapal di pelabuhan
- Number of vessel (`portTraffic()`)              : jumlah unik kapal yang masuk ke suatu pelabuhan
- Number of visit (`numVisit()`)           : jumlah kunjungan kapal di pelabuhan
- Number of draught changes (`draughtDiff(tipe='count')`)  : jumlah kapal yang mengalami perubahan draft di pelabuhan (terbagi menjadi 2, yaitu positif dan negatif)
- Draught change sizes (`draughtDiff(tipe='sum')`)       : besar perubahan draft kapal yang terjadi di pelabuhan (terbagi menjadi 2 juga)

Data indikator tersebut seperti terdapat pada folder [`out_from_geomesa/`](Sourcecode+data/out_from_geomesa/) dalam bentuk bulanan pada setiap port dan terbagi menjadi beberapa bagian (berdasarkan rentang waktu tertentu). Data tersebut akan diagregasi menjadi indikator terkait ekspor impor bulanan secara keseluruhan seperti pada file [`df_indikatorAIS.csv`](Sourcecode+data/df_indikatorAIS.csv)

# Metode Peramalan
Metode peramalan yang akan digunakan adalah model ANN dan model ARIMA. Model ARIMA akan menggunakan library `statsmodels` (`statsmodels.tsa.arima.model`). 
Sedangkan fungsi ANN akan dibangun dengan menggunakan referensi dari Coursera ([http://www.coursera.org](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/znwiG/forward-and-backward-propagation)) seperti pada file [`_2_ANN.ipynb`](Sourcecode+data/_2_ANN.ipynb).

### ANN
Tahapan ANN yang dibangun yaitu:
- Inisialisasi parameter (`initialize_params()`)
- Activation function (`fa()`)
- Forward propagation (`forward_propagation()`)
- Compute cost (`compute_cost()`)
- Backward propagation (`backward_propagation()`)
- Update parameters (`update_params()`)
- Model ANN keseluruhan (`nn_model()`)
- Predict function (`predict()`)


### ARIMA
Tahap model ARIMA (pada [`_3_exe.ipynb`](Sourcecode+data/_3_exe.ipynb)), yaitu:
- Meregresikan variabel prediktor dengan variabel respons
    ```python
      regresi = sm.OLS(Y,X).fit()
    ```
- Residual hasil regresi dibuat plot ACF dan PACF untuk menentukan parameter p dan q pada ARIMA. Penentuan parameter p dan q tersebut juga mempertimbangkan jumlah variabel yang signifikan dan error yang dihasilkan pada berbagai macam kombinasi p dan q
    ```python
      from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
      plot_acf(regresi.resid)
      plot_pacf(regresi.resid)
    ```
- Penentuan koefisien variabel / pemodelan
    ```python
      model = ARIMA(endog, order=(p,0,q), freq='MS', exog) #modelling ARIMAX
      model_fit = model.fit() #fit
      model_fit.summary() #summary
    ```
- Uji diagnostik model
    ```python
      model_fit.plot_diagnostics()
    ```
    
# Penerapan
### Variabel prediktor
Penerapan model peramalan dengan indikator AIS seperti file [`_3_exe.ipynb`](Sourcecode+data/_3_exe.ipynb). Semua indikator AIS yang terbentuk akan dijadikan variabel prediktor untuk meramalkan nilai dan volume ekspor impor. Dilakukan juga beberapa metode seleksi variabel prediktor agar memberikan hasil peramalan yang terbaik, di antaranya:
- Seleksi menggunakan permutation importance (`permutation_importance()`). Model ANN yang terbentuk (semua indikator AIS sebagai variabel prediktor) akan dilakukan permutation importance untuk melihat importance value setiap indikator. Indikator yang memiliki importance value>0 akan dipilih sebagai variabel prediktor
- Seleksi menggunakan stepwise regression (`stepwise_selection_bic()`). Indikator AIS yang memberikan peningkatan nilai BIC ke model akan dipilih sebagai variabel prediktor
- Seleksi menggunakan indikator yang signifikan (`alpha = 0.05`) pada model ARIMA yang terbentuk sebelumnya (semua indikator AIS sebagai variabel prediktor)
- Seleksi menggunakan nilai korelasi, yaitu indikator yang memiliki nilai korelasi > 0.25 dengan statistik ekspor impor

### Evaluasi peramalan
Evaluasi peramalan menggunakan:
- nilai RMSE dari library `sklearn` pada `mean_squared_error(squared=False)` 
- nilai MAPE yang dibangun sesuai rumus dengan fungsi `mape()`

### Hasil peramalan
Hasil peramalan model, baik itu ANN maupun ARIMA, dapat terlihat pada folder [`OUT`](Sourcecode+data/OUT). Folder tersebut berisi rangkuman hasil peramalan, seperti nilai RMSE, MAPE, variabel yang digunakan, parameter model yang digunakan, dll. [^2]

[^2]: Seleksi permutation importance (AISf/ann), seleksi stepwise (AISr/st), seleksi var yang signifikan arima (AISs/ar), seleksi korelasi (AISc/corr)

# Referensi
-	Achkar, R., Elias-Sleiman, F., Ezzidine, H., & Haidar, N. (2018). [Comparison of BPA-MLP and LSTM-RNN for Stocks Prediction](Sitasi/ISCBI.2018.00019.pdf). International Symposium on Computational and Business Intelligence (ISCBI).
-	Adland, R., Jia, H., & Strandenes, S. P. (2017). [Are AIS-based trade volume estimates reliable? The case of crude oil exports](Sitasi/ref%20Are_AIS-based_trade_volume_estimates_reliable_The_.pdf). Maritime Policy & Management.
-	Ahani, I. K., Salari, M., & Shadman, A. (2019). [Statistical models for multi-step-ahead forecasting of fine particular matter in urban areas](Sitasi/j.apr.2018.11.006.pdf). Atmospheric Pollution Research, 10(3), 689-700.
-	Ahmed, F., Cui, Y., Fu, Y., & Chen, W. (2021). [A Graph Neural Network Approach for Product Relationship Prediction](Sitasi/2105.05881.pdf). ASME IDETC.
-	Ahmed, N. K., Atiya, A. F., Gayar, N. E., & El-Shishiny, H. (2010). [An Empirical Comparison of Machine Learning Models for Time Series Forecasting](Sitasi/07474938.2010.481556.pdf). Econometric Reviews, 594-621.
-	Arslanalp, S., Marini, M., & Tumbarello, P. (2019, December). [Big Data on Vessel Traffic: Nowcasting Trade Flows in Real Time](Sitasi/refv%20ais%20nowcast%20trade%20flows%20wpiea2019275-print-pdf.pdf). IMF Working Paper.
-	Badan Pusat Statistik. (2019). [Statistik Perdagangan Luar Negeri Indonesia Ekspor 2019, Jilid I](https://www.bps.go.id/publication/download.html?nrbvfeve=MWZjMGY2MjUzODg0M2I1MWMyZGYyYzc5&xzmn=aHR0cHM6Ly93d3cuYnBzLmdvLmlkL3B1YmxpY2F0aW9uLzIwMjAvMDcvMDYvMWZjMGY2MjUzODg0M2I1MWMyZGYyYzc5L3N0YXRpc3Rpay1wZXJkYWdhbmdhbi1sdWFyLW5lZ2VyaS1pbmRvbmVzaWEtZWtzcG9yLS0yMDE5LS1qaWxpZC1pLmh0bWw%3D&twoadfnoarfeauf=MjAyMS0wNy0zMSAwODo0MTo1MQ%3D%3D). Jakarta: BPS RI.
-	Badan Pusat Statistik. (2020). [Ekspor Menurut Moda Transportasi Tahun 2018-2019](https://www.bps.go.id/publication/download.html?nrbvfeve=YjBlYzQ5NGIyMTU5N2IxZmE3NDE0ZDdj&xzmn=aHR0cHM6Ly93d3cuYnBzLmdvLmlkL3B1YmxpY2F0aW9uLzIwMjAvMDkvMjUvYjBlYzQ5NGIyMTU5N2IxZmE3NDE0ZDdjL2Vrc3Bvci1tZW51cnV0LW1vZGEtdHJhbnNwb3J0YXNpLXRhaHVuLTIwMTgtMjAxOS5odG1s&twoadfnoarfeauf=MjAyMS0wNy0zMSAwODo0MzoyNQ%3D%3D). Jakarta: BPS RI.
-	Badan Pusat Statistik. (2020, November). Perdagangan Luar Negeri. Diambil kembali dari Web Badan Pusat Statistik: https://www.bps.go.id
-	Badan Pusat Statistik. (2021, Februari 8). Ekspor dan Impor.Diambil kembali dari Web Badan Pusat Statistik : http://www.bps.go.id/exim/
-	Cerdeiro, D. A., Komaromi, A., Liu, Y., & Saeed, M. (2020, May). [World Seaborne Trade in Real Time: A Proof of Concept for Building AIS-based Nowcasts from Scratch](Sitasi/refv%20ais%20nowcast%20from%20scratch%20wpiea2020057-print-pdf.pdf). IMF Working Paper.
-	Chandrashekar, G., & Sahin, F. (2014). [A Survey on Feature Selection Methods](Sitasi/feature%20select%20j.compeleceng.2013.11.024.pdf). Computers and Electrical Engineering 40, 16-28.
-	Fong, S. J., Li, G., Dey, N., Crespo, R. G., & Herrera-Viedma, E. (2020). [Finding an Accurate Early Forecasting Model from Small Dataset: A Case of 2019-nCoV Novel Coronavirus Outbreak](Sitasi/2003.10776.pdf). International Journal of Interactive Multimedia and Artificial Intelligence, 6, 132-140.
-	Hammer, C., Kostroch, D. C., & Quiros, G. (2017, September). [Big Data: Potential, Challenges, and Statistical Implications](Sitasi/imf%20sdn1706-bigdata.pdf). IMF Staff Discussion Note.
-	Han, J., Kamber, M., & Pei, J. (2006). [Data mining: concepts and technique](Sitasi/book%20dm%20The-Morgan-Kaufmann-Series-in-Data-Management-Systems-Jiawei-Han-Micheline-Kamber-Jian-Pei-Data-Mining.-Concepts-and-Techniques-3rd-Edition-Morgan-Kaufmann-2011.pdf). San Fransisco: Morgan Kaufman Publisher.
-	Hanim, Y. M. (2015). [Penerapan Regresi Time Series dan ARIMAX untuk Peramalan Inflow dan Outflow Uang Kartal di Jawa Timur, DKI Jakarta, dan Nasional](https://core.ac.uk/download/pdf/291471336.pdf). Surabaya: Institut Teknologi Sepuluh Nopember.
-	International Maritime Organization. (2015, February 16). Regulations for carriage of AIS. Diambil kembali dari https://www.imo.org
-	Karlaftis, M. G., & Vlahogianni, E. I. (2011). [Statistical methods versus neural networks in transportation research: Differences, similarities and some insights](Sitasi/j.trc.2010.10.004.pdf). Transportation Research Part C: Emerging Technologies, 387-399.
-	Mason, R. D., & Lind, D. A. (1996). Teknik Statistika untuk Bisnis dan Ekonomi. Jakarta: Erlangga.
-	Mitchell, T. M. (1997). [Machine Learning](Sitasi/book%20ml.pdf). New York: McGraw-Hill.
-	Montgomery, D. C., Jennings, C. L., & Kulahci, M. (2007). [Introduction to Time Series Analysis and Forecasting](Sitasi/3%20Montgomery%20-%20Introduction%20to%20Time%20Series%20Analysis%20and%20Forecasting.pdf). Canada: John Wiley & Sons, Inc.
-	Moolayil, J. (2019). [Learn Keras for Deep Neural Networks](Sitasi/book%20Learn%20Keras%20for%20Deep%20Neural%20Networks_%20A%20Fast-Track%20Approach%20to%20Modern%20Deep%20Learning%20with%20Python%20(%20PDFDrive%20).pdf). Canada: Apress.
-	National Geospatial-Intelligence Agency. (2021, Februari 8). World Port Index: Query Results of Indonesian Port. Diambil kembali dari situs Maritime Safety Information: https://msi.nga.mil
-	NCSS Statistical Software. (2021, Juni). Stepwise Regression - Statistical Software. Diambil kembali dari ncss-wpengine.netdna-ssl.com
-	Neves, J., & Cortez, P. (1998). [Combining Genetic Algorithms, Neural Networks and Data Filtering for Time Series Forecasting](Sitasi/ga%20ann%20ts%20csc98.pdf). IMACS International Conference on Circuits, Systems and Computers (IMACS-CSC'98) (pp. 933-939). Piraeus, Greece: IMACS CSC.
-	Nooraeni, R., Sari, P. N., & Yudho, N. P. (2019). [Using Google trend data as an initial signal Indonesia unemployment rate](Sitasi/gtrend%20arima%20bsts%20bu%20rani%20ContributedPaperSessionCPS-Volume3rani.pdf). ISI Worid Statistics Congress. Kuala Lumpur.
-	Noyvirt, A. (2019). [Faster Indicators of UK Economic Activity: Shipping](Sitasi/refv%20Faster%20indicators%20of%20UK%20economic%20activity.pdf). Data Science Campus.
-	OECD. (2017). [Nowcasting Techinque to Improve Timeliness](Sitasi/improve%20timeliness%20SA-2017-7-Nowcasting-OECD.pdf). Coordination of Statistical Activities. Muscat: OECD.
-	Pramana, S., Yuniarto, B., Mariyah, S., Santoso, I., & Nooraeni, R. (2018). Data Mining dengan R Konsep Serta Implementasi. Bogor: In Media.
-	Rahkmawati, Y., Sumertajaya, I. M., & Aidi, M. N. (2019). [Evaluation of Accuracy in Identification of ARIMA Models Based on Model Selection Criteria for Inflation Forecasting with the TSClust Approach](Sitasi/yeni-ijsrp-p9355.pdf). International Journal of Scientific and Research Publications, 9(9), 439-443.
-	Raymond, E. S. (2021, Januari). AIVDM/AIVDO protocol decoding. Dipetik Juni 2021, dari https://gpsd.gitlab.io/gpsd/AIVDM.html
-	Sarwono, J. (2009). Statistik Itu Mudah: Panduan Lengkap untuk Belajar Komputasi Statistik Menggunakan SPSS 16. Yogyakarta: Universitas Atma Jaya Yogyakarta.
-	Torres, J., Avilés, D. G., Lora, A., & Álvarez, F. M. (2019). [Random Hyper-parameter Search-Based Deep Neural Network for Paper Consumption Forecasting](Sitasi/random%20search%20dlfnn%20978-3-030-20521-8_22.pdf). IWANN.
-	UN Global Working Group. (2019). [United Nations Global Platform: Data for the World](Sitasi/ais%20UNGlobalPlatform_Brochure_v1.0.1.pdf). UN Global Working Group.
-	United Nations Statistics Division. (2020, Oktober 22). AIS Handbook. Diambil kembali dari UN Statistics Wiki: https://unstats.un.org/
-	United States Coast Guard. (2021, Januari). Definition - Vessel Restricted in Her Ability to Maneuver. (USCG Navigation Center) Dipetik Juni 2021, dari https://www.navcen.uscg.gov
-	Wilcox, P. (2020). [Selecting Feature with Permutation Importance](Sitasi/Selecting%20Features%20with%20Permutation%20Importance%20-%20Lucena%20Research.pdf). Lucena Research.
-	Yang, L., & Shami, A. (2020). [On Hyperparameter Optimization of Machine Learning Algorithms: Theory and Practice](Sitasi/2007.15745.pdf). Neurocomputing.
-	Zissis, D., Xidias, E. K., & Lekkas, D. (2016). [Real-time vessel behavior prediction](Sitasi/ref%20ais%20ann%20vessel%20article.pdf). Evolving Systems(7), 29-40


# Kontak
Jimmy Nickelson <br>
221709765@stis.ac.id

---
