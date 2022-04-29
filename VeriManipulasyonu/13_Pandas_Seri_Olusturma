import pandas as pd

seri = pd.Series([10,88,3,4,5])
print(seri)
print("---------------------------")
print(type(seri))                       # Tipi
print("---------------------------")
print(seri.axes)                        # Index bilgileri
print("---------------------------")
print(seri.dtype)
print("---------------------------")
print(seri.size)                      # büyüklük 5
print("---------------------------")
print(seri.ndim)                      # Boyut 1 boyutlu
print("---------------------------")
print(seri.values)                      # Değerler
print("---------------------------")
print(seri.head())                      # İlk 5 veri
print("---------------------------")
print(seri.tail())                      # Son 5 veri
print("---------------------------")

### Index değiştirme

a = pd.Series([99,22,332,94,32], index=["a","b","c","d","e"])
print(a)
print("---------------------------")
print(a["a"])                   # Bu şekilde ındex çağırabiliriz
print("---------------------------")

### SÖZLÜK ÜZERİNDEN PD OLUŞTURMA

sozluk = pd.Series({"reg":10, "log":11, "cart":12})
print(sozluk)
print("---------------------------")
print(sozluk["reg":"log"])              # Bu şekilde çağırabiliyoruz.
print("---------------------------")
print(pd.concat([seri,seri]))           # Bu şekilde birleştirme de yapabiliyoruz.
