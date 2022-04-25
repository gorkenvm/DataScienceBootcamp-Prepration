import numpy as np

a = np.array([1,2,3,4,5])
print(a)
print("----------------------------------------")
print(type(a))
print("----------------------------------------")
print(np.array([3.15,5,2,6,13])) # float girdiğimiz için hepsini float yaptı. Tek tip olması için
print("----------------------------------------")
####################SIFIRDAN ARRAY OLUŞTURMA ###################################

sifir = np.zeros(10, dtype = int) # 10 tane sıfır oluştur, tipi int olsun.
print(sifir)
print("----------------------------------------")
bir = np.ones((3,5), dtype = int)  # Matris şeklinde oluştur.
bir2 = np.ones(5, dtype = int)     # 1 boyutlu oluştur.
print(bir)
print("----------------------------------------")
print(bir2)
print("----------------------------------------")
ful = np.full((3,5), 3)             # 3'e 5 matris oluştur. 3 yaz içine
print(ful)
print("----------------------------------------")
print(np.full(5,3))                 # 5 tane oluştur. 3 yaz içine
print("----------------------------------------")
print(np.linspace(0,1,10))          # 0'dan 1'e kadar 10 tane oluştur
print("----------------------------------------")
print(np.random.normal(10,4,(3,4)))     # mean=10, std=4, (3,4) matris
print("----------------------------------------")
print(np.random.randint(0,10, (3,3)))   # 0'dan 10'a kadar (3,3) matris oluştur.



