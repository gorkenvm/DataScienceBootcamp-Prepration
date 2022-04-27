import numpy as np

v = np.array([2,1,4,3,5])
print(np.sort(v))           # Array sıralamada BUNU KULLAN.
print("--------------------------")
print(v.sort())         # Bunu Kullanma verinin yapısını bozuyor.
print("--------------------------")

### İKİ BOYUTLU

m = np.random.normal(20,5, (3,3))
print(m)
print("--------------------------")
sa = np.sort(m, axis= 1)              #Satıra göre sıralama
print(sa)
print("--------------------------")
su = np.sort(m, axis= 0)                #Sutuna gore sıralama
print(su)


