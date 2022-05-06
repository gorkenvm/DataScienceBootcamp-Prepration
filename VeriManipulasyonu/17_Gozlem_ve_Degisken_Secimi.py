import numpy as np
import pandas as pd
### LOC & İLOC
m = np.random.randint(1,30, size=(10,3))
df = pd.DataFrame(m, columns=["var1","var2","var3"])
print(df)
print("---------------------------")
### loc: tanımlandığı şekli ile seçim yapmak için kullanılır.
print(df.loc[0:3])      # loc'ta 0:3 0dan3e kadar 3dahil demektir.
print("---------------------------")
### iloc: alışık olduğumuz indeksleme mantığı ile seçim yapar.
print(df.iloc[0:3])     # 0dan 3e kadar. 3 dahil değildir.
print("---------------------------")
print(df.iloc[0,0])
print("---------------------------")
print(df.iloc[:3, :2])          # 3 satır, 2 sutün
print("---------------------------")
print(df.loc[0:3, "var3"])      # iloc'ta int str bir arada kullanınca hata veriyor.
print("---------------------------")
print(df.iloc[0:3]["var3"])     # hata almamak için yanına yazıyoruz column'ı
print("---------------------------")




















