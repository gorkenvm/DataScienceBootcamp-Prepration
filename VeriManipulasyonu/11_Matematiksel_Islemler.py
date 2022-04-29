import numpy as np

v = np.array([1,2,3,4,5])
print(v-1)          # - + * / ** gibi tüm işlemler yapılabiliyor.
# Bu işlemler arka tarafta
# ufunc denilen mekanizma çalıştıgı için oluyor.
# np.substract(v, 1) - dediğimizde bu işi yapar.
# n.divide(), np.add(), np.multiply() gibi işlemlerin ismi ufunc'tır.
# ama biz + - * gibi operatörler ile yapabiliyoruz.

print(np.absolute(np.array([-3])))      # Mutlak değer
print(np.cos(180))
print(np.log(v))

## CHEATSHEET lerde birsürü fonksiyon alıp deneyebiliriz.

