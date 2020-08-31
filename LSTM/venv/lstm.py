import itertools
from math import sqrt
from datetime import datetime
from numpy import concatenate
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers.recurrent import LSTM
from sklearn.utils import shuffle
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

#verimizi data.csv dosyasından okuduk
data=pd.read_csv('data2.csv')

#datamızdaki confirmed verisinden iyileşen ve ölen kişi sayısını çıkardık (anlık aktif corona sayısı için)
data['Confirmed']= data['Confirmed'] -( data['Recovered']+ data['Deaths'] )
#index resetledik
data= data.reset_index()['Confirmed']
#ilk 5 veriyi bastık
print(data.head())


#0-1 arasında Minimum Maksimum Ölçeklendirme Kullanarak Normalizasyon işlemi yapıyoruz
#Bu işlem verinin dağılımını görmemizi sağlıyor, ayrıyeten model fitlerken zaman ve performans kazandırıyor
scaler = MinMaxScaler(feature_range=(0,1))
#1 boyutlu diziye çeviriyor ve normalizasyon işlemi yapıyor
data=scaler.fit_transform(np.array(data).reshape(-1,1))

# training_size'a dinamik olarak verinin %60'ının uzunluğunu atadık
training_size=int(len(data)*0.60)
# test_size'a dinamik olarak %40'ının uzunluğunu atadık
test_size=len(data)-training_size

#train_data'ya datanın baştan %60'ına kadar olan kısmını atadık
#test_data'ya datanın kalan %40'ını atadık
train_data,test_data=data[0:training_size,:],data[training_size:len(data),:1]

# bu fonksiyon, time_step sayısı kadar veriyi x_train'e atıyor... [Veri1,Veri2,Veri3]
# time_step+1 verisini de y_train'e atıyor.. [Veri4]
def create_dataset(dataset, time_step=1):
    dataX, dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a= dataset[i:(i+time_step),0] # i=0, 0,1,2,3
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)

time_step= 3  # optimizasyon sebepleriyle değişkenlik gösterebilir

#üstteki fonksiyona gönderiliyor
x_train , y_train= create_dataset(train_data,time_step)
x_test , y_test= create_dataset(test_data,time_step)

print(y_train)

# input'u yeniden şekillendiriyoruz [samples, time steps, features] LSTM MODELİ BÖYLE İSTİYOR
x_train= x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test= x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#Modelimizi Tek katmanlı , 50 unitli ve dropout'u 0.2 alarak yaptık
#Tek Katmanlı LSTM kullanmamızın sebebi over fitting'den kaçınma
model = Sequential()
model.add(LSTM(units = 50, return_sequences = False, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Çıkış Katmanımız
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#modelimizin özeti
model.summary()

#modelimizi eğitim verimizle (x_train ve y_train) ve test verimizle (x_test ve y_test) ile , 100 kez döngüyle fitliyoruz
model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=100,verbose=1)


#model.predict() fonksiyonu ile x_train verisiyle tahmin yapıyoruz
train_predict=model.predict(x_train)
#model.predict() fonksiyonu ile x_test verisiyle tahmin yapıyoruz
test_predict=model.predict(x_test)

#inverse işlemi yapıyoruz (unnormalizasyon)
train_predict=scaler.inverse_transform(train_predict)
#inverse işlemi yapıyoruz (unnormalizasyon)
test_predict=scaler.inverse_transform(test_predict)

#######
#verimizin RMSE(Root Mean Square Devination) değerine bakıyoruz.
#bir model veya bir tahminci tarafından öngörülen değerler ile gözlenen değerler arasındaki farklar
#print(math.sqrt(mean_squared_error(y_train,train_predict)))
#print(math.sqrt(mean_squared_error(y_test,test_predict)))
#######



#Bu kod yalnızca; test tahminini, Plotlibde doğru zamanında göstermek için yapıldı...
look_back=time_step
testPredictPlot=np.empty_like(data)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(data)-1,:]=test_predict

##Plotlib ile figürleştirme
fig= plt.figure(figsize=(15,8))
plt.plot(scaler.inverse_transform(data), label="Gerçek Veri", color="orange")
plt.plot(train_predict,color = "blue", label = "Eğitim Tahmini")
plt.plot(testPredictPlot, color = "red", label = "Test Tahmini")
plt.title('COVID-19: Eğitim-Test Tahmini')
plt.xlabel('Günler')
plt.legend(loc='best')
plt.savefig("COVID-19 Train-Test")
plt.show()

#x_input'un boyutunu test_data-time_step olarak yaratıyoruz (test_data-3) sonra da elimizdeki test_data'yı buraya atıyoruz
x_input=test_data[len(test_data)-time_step:].reshape(1,-1)

#tüm x_input'u listeye çevirip temp_input'a atıyoruz
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


lst_output = []

#buraya time_step gelmeli (LSTM ALGORİTMA MANTIĞINA GÖRE)
n_steps = time_step

#tahmin edilecek gün sayısı
predictDayNumber=30
i = 0
# input = (gün1 verisi, gün2verisi, gün3verisi) output=(gün4tahmini)
# sonrasında yhat'a atanıyor o da lst_output'a ekleniyor
#print olarak bastırıyorum inceleyebilirsin
while (i < predictDayNumber):

    if (len(temp_input) > n_steps):
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i + 1

lst_output2=scaler.inverse_transform(lst_output)
print(lst_output2)

#buradaki lst_output  input(d1,d2,d3) output(d4)
#                     input(d2,d3,d4) output(d5).....
#                     input(d31,d32,d33) output(d34) olarak önceki verileri göz önüne alarak, daha sonrasında da kendi tahminlerini
#                     göz önüne alarak kendi kendine veri sağlamakta


day_new=np.arange(1,165)
day_pred=np.arange(165,195)


#Plotlib ile figürleştirme
fig= plt.figure(figsize=(15,8))
df=data.tolist()  #163 günlük veri

plt.plot(day_new,scaler.inverse_transform(df), color="orange", label = "Veri Setimizdeki veriler")
plt.plot(day_pred,scaler.inverse_transform(lst_output), color="red", label="30 Günlük Tahmin")

plt.title('COVID-19: LSTM ile 1 Aylık Veri Tahmini ')
plt.legend(loc='best')
plt.xlabel('Günler')
plt.savefig("30 Gün Tahminli Toplam Veri")
plt.show()


#Plotlib ile figürleştirme
fig= plt.figure(figsize=(15,8))
df=data.tolist()  #163 günlük veri

plt.plot(day_pred,scaler.inverse_transform(lst_output), color="red", label="30 Günlük Tahmin")

plt.title('Yalnızca 30 günlük tahmin verisi')
plt.legend(loc='best')
plt.xlabel('Günler')
plt.savefig("30 Gün Tahmin")
plt.show()