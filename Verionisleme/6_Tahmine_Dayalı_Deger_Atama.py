import seaborn as sns
import missingno as msno
import numpy as np
import pandas as pd
from ycimpute.imputer import knnimput, EM
from ycimpute.imputer import MIDA

df = sns.load_dataset('titanic')
df = df.select_dtypes(include= ['float64', 'int64'])
print(df.head())
print(df.isnull().sum())    # Eksik değerlerin toplamı


### Öncelikle ML metodu kullanacağımız için df'i array'e dönüştüreceğiz.
## Daha sonra df'e çevirken için column name'leri saklamamız lazım.

var_names = list(df)    # column name'leri saklıyoruz.
n_df = np.array(df)     # df'i np array'e çevirdik.
print(n_df[0:10])       # ilk 10 gözleme bakalım.
print(n_df.shape)       # 891 row 6 column vardır.

#####  KNN İLE DOLDURMA

dff = knnimput.KNN(k = 4).complete(n_df) # dff adında nesne oluştur.
# KNN algoritması ile komşuluk sayısı 4 olsun, n_df array'ini doldur.
dff = pd.DataFrame(dff, columns= var_names) # dff'i dataframe'e dönüştür. columns name'leri var_names'ten al.
print(dff.isnull().sum()) # Eksik değerleri gözlemledik hepsi doldurulmuş.

####   EM ( Expectation Maximization) İLE DOLDURMA
df = sns.load_dataset('titanic')
df = df.select_dtypes(include= ['float64', 'int64'])
var_names = list(df)    # column name'leri saklıyoruz.
n_df = np.array(df)     # df'i np array'e çevirdik.
# Verileri sıfırladık şimdi Random Forest uygulayacağız.
dff = EM().complete(n_df)
dff = pd.DataFrame(dff, columns= var_names) # dff'i dataframe'e dönüştür. columns name'leri var_names'ten al.
print(dff.isnull().sum()) # Eksik değerleri gözlemledik hepsi doldurulmuş.


# ÖNERİ// Tahmine dayalı doldurma yapacaksak bile; diğer değişkenlere baglılık var mı yani yapısal bir sorun var mı kontrol etmemiz lazım.

## GAIN, MIDA, MICE metodlarını araştırıp bunlarile de imput edebilirsin.











