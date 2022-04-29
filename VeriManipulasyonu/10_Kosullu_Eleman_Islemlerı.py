import numpy as np

v = np.array([1,2,3,4,5])
print(v<3)                  # Kucuk olanlara TRUE, Kucuk olmayanlara FALSE dedı
# Bunu da Fancy ile yakalayacagız.

print("---------------------")
print(v[v<3])               # V nin içine girdi, True olanların değerini getirdi.
print("---------------------")
print(v[v>=3])               # V nin içine girdi, True olanların değerini getirdi.
print("---------------------")
print(v[v<=3])               # V nin içine girdi, True olanların değerini getirdi.
print("---------------------")
print(v[v==3])               # V nin içine girdi, True olanların değerini getirdi.
print("---------------------")
print(v[v!=3])               # V nin içine girdi, True olanların değerini getirdi.
print("---------------------")
# Vektorel calısabıldıgımız ıcın vektorel operasyonlar yapabılırız
print(v*2)                      # Hepsini 2 ile çarpar, diğer matematiksel işlemlerde oluyor.
print("---------------------")






