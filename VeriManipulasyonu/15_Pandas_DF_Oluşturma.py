import pandas as pd

l = [1,2,39,67,90]
df = pd.DataFrame(l, columns=["degisken_ismi"])     # DF oluşturduk.
print(df)
print("---------------------------------")
import numpy as np
m = np.arange(1,10).reshape((3,3))
print(m)
print("---------------------------------")
dfm = pd.DataFrame(m,columns=["var1","var2","var3"])
print(dfm)
print("---------------------------------")
### DF İSİMLENDİRME
print(dfm)
print("---------------------------------")
dfm.columns = ["deg1","deg2","deg3"]        # columns isimleri değişti
print(dfm)
print("---------------------------------")
print(type(dfm))            # Pandas.core.frame.dataframe
print("---------------------------------")
print(dfm.axes) # satır sütun bilgilerini verir. Index 1'den başlar 3'e kadar gider 1 er 1er artar.
print("---------------------------------")
print(dfm.shape)    # boyut bilgisi
print("---------------------------------")
print(dfm.ndim)     # kaç boyutlu
print("---------------------------------")
print(dfm.size)     # kaç eleman var
print("---------------------------------")
print(dfm.values)   # verileri array'e dönüştürüp bize gösteriyor.
print("---------------------------------")
print(dfm.head())       # ilk 5
print("---------------------------------")
print(dfm.tail())       # son 5 veriyi gösterir.
print("---------------------------------")

a = np.array([1,2,3,4,5])
pd.DataFrame(a, columns= ["deg1"])  # yeni bir df oluşturduk.
print(a)




























