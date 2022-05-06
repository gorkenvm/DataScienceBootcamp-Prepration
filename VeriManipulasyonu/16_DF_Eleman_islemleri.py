import pandas as pd
import numpy as np
s1 = np.random.randint(10, size = 5)
s2 = np.random.randint(10, size = 5)
s3 = np.random.randint(10, size = 5)

sozluk = {"var1":s1, "var2":s2, "var3":s3}  # Sözlüğe array yerleştirdik.
df = pd.DataFrame(sozluk)
print(df)           # Sözlük üzerinden df oluşturduk.
print("---------------------------")
print(df[0:1])      # slicing
print("---------------------------")
print(df.index)     # indexleri görüntüle
print("---------------------------")
df.index = ["a","b","c","d","e"]        # index değiştirdik.
print("---------------------------")
print(df["c":"e"])  # index seçme
print("---------------------------")
#### SİLME
df.drop("a", axis=0, inplace=True)    # o satırı silmek istediğimizi axis=0 ile söylüyoruz. İnplace=True dersek uygular yoksa sadece görüntüleriz.
print(df)
print("---------------------------")
df.drop(["c","d"], axis = 0, inplace=True) # Tek tek silmek zorunda değiliz
print(df)
print("---------------------------")
print("var1" in  df)
print("---------------------------")
l = ["var1", "var2", "var4"]
for i in l:             # l ' dek, değerler df nin içinde mi ?
    print(i in df)

df["var4"] = df["var1"]/df["var2"]   # df["var4"] olmadığı için, olultur dediğimizi zannediyor. var1/var2 sonuçlarını yaz oraya diyoruz.
print(df)
print("---------------------------")
## Değisken silme
df.drop("var4", axis=1, inplace=True)  # axis = 1 column olarak sil demektir.
print(df)
print("---------------------------")
l = ["var1", "var2"]
print(df.drop(l, axis=1))   # sildik ama kaydetmedik sadece görüntüledik.





























