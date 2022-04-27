import numpy as np

x = np.array([1,2,3,99,99,3,2,1]) # Arrayi böleceğiz
a,b,c = np.split(x, [3,5])   # 3'e kadar, 5'e kadar ve 5'ten sonrası diye böl
print(a,'\n',b,'\n',c)
print("-------------------------------")
##İKİ BOYUTLU

m = np.arange(16).reshape(4,4)
print(m)
print("-------------------------------")
ust, alt = np.vsplit(m,[2])             # Yatayda 2ye kadar böl
print(ust)
print("-------------------------------")
print(alt)
print("-------------------------------")
sag, sol = np.hsplit(m,[2])
print(sag)
print("---------------------------------")
print(sol)