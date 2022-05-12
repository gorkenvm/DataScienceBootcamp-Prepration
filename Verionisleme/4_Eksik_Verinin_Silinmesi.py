import numpy as np
import pandas as pd

V1 = np.array([1, 3, 6, np.NaN,7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN,5,8,12,np.NaN, np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
df = pd.DataFrame(
    {"V1" : V1,
     "V2" : V2,
     "V3" : V3}
)
print(df)

print("----------------------")
print(df[df.notnull().all(axis=1)])
print("----------------------")
# SİLME YÖNTEMLERİ

df.dropna() # En az 1 eksik değer bile varsa o gözlemleri sil.
df.dropna(how= "all")   # Hepsi aynı anda eksik ise o gözlemi sil.
df.dropna(axis= 1)      # Bu gözlem bazında değilde değişken(Sütun) bazında silecek. Her değişkende en az 1 tane NaN olduğu için hepsini sildi.
df.dropna(axis= 1, how= "all") # Sütun bazında tüm değişkeneleri NaN olan değişkeni sil, böyle bir sütun olmadığı için silme işlemi yapmadı.
df["sil_beni"] = np.nan     # Tüm değerleri nan olan bir değişken oluşturduk.
print(df)
df.dropna(axis= 1, how= "all", inplace= True) # Tekrar çalıştırdık.
print(df)                           # Görüldüğü gibi "sil_beni" değişkeni silindi.

