### LSTM-COVID-19 Projesi nedir?

Bu proje Türkiye Korona Virüs dataseti üzerinden (11/3/2020 - 21/08/2020), LSTM ve GRU algoritmalarını kullanarak  22/08/2020 - 20/09/2020 tarihleri arasındaki olası aktif korona sayısını tahmin etmekte.
Bu işlem sırasında LSTM algoritmasında, önceki 3 veriyi okuyarak sonraki veriyi tahmin etmekte. (Örneğin; 19,20,21 ağustostaki aktif korona sayısını kullanarak 22 ağustostaki aktif korona sayısını tahmin eder.)  

**Bu Projede**

- MinMaxScaler fonksiyonu ile ölçeklendirme işlemimizi yaptık.
- eğitim verimiz , tüm verimizin %60'ı olarak seçildi.
- test verimiz, tüm verimizin %40'ı olarak seçildi.
- Timstep'imiz 3 olarak seçildi.
- LSTM tek katmanlı ve 50 hücreli seçildi.
- Overfitting sıkıntısını önlemek adına dropout=0,2 olarak seçildi.
- optimizer'ımız 'adam' , loss değerimiz 'mse' seçildi
- epoch değerimiz 100 ayarlandı
- dinamik fonksiyonlar yaratıldığından, predictDayNumber değişkeni değiştirilerek , tahmin sayısı arttırılabilir.


### --------------------

### Kurulum

### 1. Repo'yu Klonla
```sh
git clone https://github.com/halilkolge/LSTM-COVID-19.git
```
### 2. Pip paketlerini indir
```sh
pip install numpy
pip install pandas
pip install sklearn
pip install keras
pip install seaborn
pip install matplotlib
pip install tensorflow
```

### --------------------
