import numpy as np
import pandas as pd

m = np.random.randint(1,30, size=(5,3))
df1 = pd.DataFrame(m, columns=["var1","var2","var3"])

df2 = df1 +99           # df1 e 99 ekleyip yeni bir df oluşturduk.

print(pd.concat([df1,df2])) # dikkat ettiyseniz indexler 1den4e kadar ve 1# den4e kadar
print("-----------------------------")
print(pd.concat([df1,df2], ignore_index= True)) # ignore_index ile bu sorunu çözebiliyoruz.
print("-----------------------------")
print(df1.columns)
df2.columns = ["var1", "var2", "deg3"]  # column ismi değiştirildi.
print("-----------------------------")
print(pd.concat([df1,df2])) # var1,var2,var3   ve var1,var2,deg3  farklı columlar var oyüzden birleştirmede sorun yaşıyourz.
print("-----------------------------")
print(pd.concat([df1,df2], join="inner")) # inner join ile kesişimlerini birleştirdik.
print("-----------------------------")
























