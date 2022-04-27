import numpy as np

v =  np.arange(0, 30, 3)  # 0 dan 30 a kadar 3 er 3 er artarak git
print(v)
print("---------------------")
# Normalde Aşağıdaki gibi erişiyoruz.
print(v[1])
print("---------------------")

# Fancy ile

al_getir = [1,3,5]  # liste olarak kaydediyoruz çağırmak istediklerimizi
print(v[al_getir])  # içine yazılan indeksleri tek tek getiriyor.
print("---------------------")

### İKİ BOYUTTA
m = np.arange(9).reshape((3,3))
print(m)
print("---------------------")
satir = np.array([0,1])
sutun = np.array([1,2])
print(m[satir,sutun])           # Oluşturulan satır sutun içindeki degerleri içine alıp çalışıyor.
print("---------------------")
print(m[0:, [1,2]])             # satır normal slicing, sutun ise fancy ve direkt yazıldı, birşeye atamak zorunda degılız
print("---------------------")





