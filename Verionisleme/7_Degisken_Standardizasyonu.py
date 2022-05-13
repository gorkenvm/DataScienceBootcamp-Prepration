# Değişken Standardizasyonda, yayılım dagılım değişmeyecektir.
# Değişken Dönüştürmede, yayılım da değişiyor.

import numpy as np
import pandas as pd
V1 = np.array([1,3,6,5,7])
V2 = np.array([7,7,5,8,12])
V3 = np.array([6,12,5,6,14])
df = pd.DataFrame(
    {"V1" : V1,
     "V2" : V2,
     "V3" : V3})
df = df.astype(float)
print(df)

##### STANDARDİZASYON
from sklearn import preprocessing
preprocessing.scale(df)
print(preprocessing.scale(df))  # Hepsinde standart bir dönüşüm yaptıgından, birbirleriyle kıyaslanabilirliğinde değişim olmadı
print(df)                       # Görüldüğü üzere yapısında da değişiklik yok, copy şeklinde işlem yapıyor.
##### NORMALİZASYON
# tüm değerleri 0 - 1 arasına dönüştürür.
print(preprocessing.normalize(df))

##### Min-Max DÖNÜŞÜMÜ
# istediğimiz iki değer arasına dönüştürmek istediğimizde
scaler = preprocessing.MinMaxScaler(feature_range= (10,20)) # Ölçek oluşturduk, 10 - 20 arası olsun dedik
scaler.fit_transform(df)                                    # ölçek kullanarak df'i dönüştürdük
print(scaler.fit_transform(df))                             # Burada sadece print ettik