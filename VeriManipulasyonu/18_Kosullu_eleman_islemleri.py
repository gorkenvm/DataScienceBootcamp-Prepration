import numpy as np
import pandas as pd

m = np.random.randint(1,30, size=(10,3))
df = pd.DataFrame(m, columns=["var1","var2","var3"])

print(df[df.var1 > 15]) # var1 15 ten büyük olanları getir.
print("----------------------")
print(df[df.var1 > 15]["var2"]) # yukarıdaki ifade df veriyor. yanına "var2 yazınca o column'ı veriyor
print("----------------------")
print(df[(df.var1 > 15) & (df.var3 < 5)]) # var1 > 15 ve var3 < 5 olanları seç.
print("----------------------")
print(df[(df.var1 > 15)][["var1","var2"]])  # koşulda column seçtik.
print("----------------------")
