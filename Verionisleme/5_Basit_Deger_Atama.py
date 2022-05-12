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

############ DEĞER ATAMA YÖNTEMLERİ ##################
#######SAYISAL DEĞİŞKENLERDE ATAMA ###################

df["V1"].fillna(0) # V1 deki tüm eksik verilere 0 ata
df["V1"].fillna(df["V1"].mean()) # V1 deki tüm eksik verilere V1 verilerinin ortalamasını ata
# TÜM DEĞİŞKENLER İÇİN BİRİNCİ YOL
df.apply(lambda  x: x.fillna(x.mean()), axis= 0)
# TÜM DEĞİŞKENLER İÇİN İKİNCİ YOL
df.fillna(df.mean()[:]) # Üsttekiyle aynı sonucu verir.

# Mesela, Dağılımlarına baktık ve V1 ve V2 değişkenleri normal dağılım ozmn mean ile doldurulabilir
# Fakat V3 değişkeni çarpık ozmn onu mean mantıklı olmayacaktır. veya median ile dolduralım dedik.
df.fillna(df.mean()["V1":"V2"]) # V1 ve V2 yi kendi ortalamaları ile doldur.
df["V3"].fillna(df["V3"].median())  # V3 ü kendi median değeri ile doldur.
# TÜM DEĞİŞKENLER İÇİN ÜÇÜNCÜ YOL
df.where(pd.notna(df), df.mean(), axis= "columns")

#######KATEGORİK DEĞİŞKENLERDE ATAMA ###################
# Ör// Maaş kolonu, tüm maaşlara göre ortalama yaparsak hata yapmış oluruz.
# Departman'a göre kırılım yapıp, o departmanın ortalamasına atamak çok daha mantıklı olacaktır.
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4 = np.array(["IT","IT","IK","IK","IK","IK","IK","IT","IT",])

df = pd.DataFrame(
    {"maas": V1,
     "V2": V2,
     "V3": V3,
     "departman": V4}
)
print(df)

print(df.groupby("departman")["maas"].mean())   # Departmanlara göre ortalama aldık

df["maas"].fillna(df.groupby("departman")["maas"].transform("mean"))
# fillna; boş olanları doldur. groupby departmana göre grupla maaşları al, transform; ortalama uygula

V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V4 = np.array(["IT",np.nan,"IK","IK","IK","IK","IK","IT","IT"], dtype= object)

df = pd.DataFrame(
    {"maas": V1,
     "departman": V4}
)
print(df)
# Kategorik işlemlerde en pratik ve verimli yol mod işlemidir.
print("------------------")
print(df["departman"].mode()[0]) # mod alınca IK geliyor, sadece string olarak almak için [0] ekledik.
df["departman"].fillna(df["departman"].mode()[0])   # Departman modu ile departmandaki nsn leri doldur dedik. Kalıcı olması için inplace= True yapmak lazım bunu artık biliyoruz.

df["departman"].fillna(method = "bfill") # Kendinden sonraki değerle doldur.
df["departman"].fillna(method = "ffill") # Kendinden önceki değerle doldur.













