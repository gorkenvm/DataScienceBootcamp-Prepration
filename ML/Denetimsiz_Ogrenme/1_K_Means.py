# Amaç gözlemleri birbirine olan benzerliklerine göre kümelere ayırmaktır.
# Oluşturulan kümelerin kendi içinde homojen , birbirlerine göre hetorojen olması beklenir
# Yani amaç ; Küme içi benzerlik maksimum, kümeler arası benzerlikte minimum yapmaktır.
# Adım 1 : Küme Sayısı Belirlenir.
# Adım 2 : Rastgele k merkez seçilir
# Adım 3 : Her gözlem için merkezlere uzaklıklar hesaplanır
# Adım 4 : Her gözlem en yakın olduğu merkeze yani kümeye atanır
# Adım 5 : Kümeler için tekrar merkez hesaplamaları yapılır.
# Adım 6 : Bu işlem belirlenen bir iterasyon adedince tekrar edilir.
# Kareler toplamlarının minimum olduğu durumdaki gözlemlerin kümelenme yapısı nihai kümelenme olarak seçilir.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# VERİ SETİ
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\USArrests.csv", index_col= 0)
print(df.head())
# Murder : Cinayet
# Assault : Saldırı
# Urbanpop : Bölgenin nufusu
# Rape     : Taciz
# AMAÇ : Eyaletleri kümelere ayırarak nerede ne kadar suç işlendiğini görmek ve buna göre yasalar oluşturmak.
print(df.isnull().sum())
print(df.info())
print(df.describe().T)

#df.hist(figsize=(10,10))
#plt.show()

kmeans = KMeans(n_clusters= 4)  # 4 sınıflı bir kmeans oluşturuldu.
k_fit = kmeans.fit(df)          # fit edildi
print(k_fit.n_clusters)         # Gözlemlerin özelliklerine bu şekilde erişebiliyoruz
print(k_fit.cluster_centers_)
print(k_fit.labels_)            # Hangi cluster'lara ait olduğunu gördük.

# KÜMELERİN GÖRSELLEŞTİRİLMESİ
k_means = KMeans(n_clusters=2).fit(df)
kumeler = k_means.labels_              # basit olsun diye 2 cluster yapıp labellerı assign ettik
plt.scatter(df.iloc[:,0], df.iloc[:,1], c = kumeler, s = 50, cmap = 'viridis')


merkezler = k_means.cluster_centers_
plt.scatter(merkezler[:,0], merkezler[:,1], c = 'black', s = 200, alpha=0.5)
plt.show()







