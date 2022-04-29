import numpy as np
import pandas as pd

a = np.array([1,2,33,444,75])   # seri için array oluşturduk.
seri = pd.Series(a)
print(seri)                     # array kullanarak seri oluşturduk.
print("---------------------------------")

print(seri[0])      # index seçme
print("---------------------------------")
print(seri[0:3])           # seri de slicing
print("---------------------------------")
seri = pd.Series([121,200,150,99], index = ["reg", "log", "cart", "rf"]) # Yeni seri oluşturduk ve indexleri ayarladık.
print(seri.index)           # serinin indexini çağırdık
print("---------------------------------")
print(seri.keys)            # serinin anahtarlarını çağırdık. index ve item le.
print("---------------------------------")
print(list(seri.items()))      # serinin içeriğini alarak listeye dönüştürdük
print("---------------------------------")
print(seri.values)              # sadece değerleri aldık.
print("---------------------------------")
### Eleman sorgulama
print("reg" in seri)            # var ise TRUE döndürecek.
print("---------------------------------")
print("a" in seri)              # olmadığı içi FALSE döndürecek.
print("---------------------------------")
print(seri["reg"])              # index ile eleman çağırma.
print("---------------------------------")
#### FANCY
print(seri[["rf", "reg"]])
print("---------------------------------")
seri["reg"] = 130               # Değer atama yapabiliyoruz.
print(seri["reg"])              # atadığımız değer başarılı
print("---------------------------------")
print(seri["reg":"log"])        # Slice işlemi yapabiliyoruz.








































