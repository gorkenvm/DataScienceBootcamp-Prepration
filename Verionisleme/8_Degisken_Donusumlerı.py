# Kafa karışıklıgı olmaması için Değişken Dönüşümü > Değişken Standardizasyon'u
# Yani standardizasyonlar da birer dönüşüm işlemidir.
### Önemli olan değişkenlerin mevcut taşıdıgı bilginin yapısını bozuyor muyuz bozmuyor muyuz. Bunu bilmek lazım
import pandas as pd
import seaborn as sns
import numpy as np
df = sns.load_dataset("tips")
df.head()

######### 0 - 1 DÖNÜŞÜMÜ
# Kategorik bir veriyi 0 1 olarak değiştirmek istediğimizde kullanıyoruz.
# Mesela Male Female 0 1 şeklinde dönüştürüyoruz.
# Bunun için En sık kullanılan yöntem LabelEncoder yöntemidir.
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()    # Dönüştürücüyü oluşturduk.
print(lbe.fit_transform(df["sex"]))   # fit edip, dönüştürmek istediğimiz kategorik değişkeni yazıyoruz.
df["yeni_sex"] = lbe.fit_transform(df["sex"])   # Yeni bir değişkene atıyoruz. female'e 0, male'e 1 dedi
print(df)
# 0 neye 1 neye geliyor bilmek önemli!!! ilgilendiğimiz değişkeni 1 yapmayı tercih ediyoruz.
# ÖR// titanic veri setinde hayatta kalanlar 1 ölenler 0, eğer hayatta kalanlar ile ilgileniyorsak.

######## 1 VE DİĞERLERİ(0) DÖNÜŞÜMÜ
df["day"].str.contains("Sun")   # Bunu iyi anlayalım, day içindekileri str yap, Sun içerenleri ver bana TRUE FALSE olarak
np.where(df["day"].str.contains("Sun"), 1, 0) # Where(x, 1, 0) x istediğimiz veri, x'i 0 yap, diğerlerini 1 yap demek where
# Yukarıyı anlamak için yazdık, aşağıdaki gibi tek satırda hallediliyor aslında
df["yeni_day"] = np.where(df["day"].str.contains("Sun"), 1, 0)  # Sun olanları 1, diğerlerini 0 yap dedik
print(df)

######## ÇOK SINIFLI DÖNÜŞÜM
print(lbe.fit_transform(df["day"])) # lbe yukarıda oluşturulmuştu. day içindekiler rakamlarla kategorize edildi
# Çooook Dikkat
# Algoritmaya bunu verince 0 1 2 3 gibi sayısal değer zannedecek o yüzden yanlış işlem yapmış olacak.
# Peki bunların kategorik oldugunu nasıl anlatacaz algoritmaya, One Hot Encoding Yapacagızç

#### One-Hot Dönüşümü ve Dummy Değişken Tuzağı
df_one_hot = pd.get_dummies(df, columns=["sex"], prefix= ["sex"])   # prefix, ön isimlendirme demektir.
print(df_one_hot)
# iki değişkenli olanlar için bunu yapmak mantıklı değil, bundan kaçınmamız lazım.
print(pd.get_dummies(df, columns= ["day"], prefix= ["day"]))
# Bu işlemin bize önemli 2 faydası vardır.
## 1 kategoriyi numerige dönüştürüyor
## 2 kategorik verinin içinde ki veriden hangisinin ağırlıgı oldugunu görebiliyoruz.
# Dönüştürme işlemleri arasında en önemli olanıdır dummy.













