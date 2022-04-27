import numpy as np

a = np.arange(20,30)
print(a)
print("--------------------------------------")
print(a[0:3])           # 0'dan 3'e kadar
print("--------------------------------------")
print(a[3:])            # 3'ten Sona kadar
print("--------------------------------------")
print(a[1::2])          # 1 den başla 2 şer 2şer git
print("--------------------------------------")
print(a[0::3])          # 0 dan başla 3 er git
print("--------------------------------------")

## İKİ BOYUTLU

m = np.random.randint(10, size=(5,5))
print(m)
print("--------------------------------------")
print(m[:,0])               # Tüm satırlar 0. sutun
print("--------------------------------------")
print(m[0,:])               # 0. satır tüm sutünlar
print("--------------------------------------")
print(m[1:3,2:4])           # 1. satırdan 3. satıra kadar, 2. sutünden 4. sutüna kadar
print("--------------------------------------")




