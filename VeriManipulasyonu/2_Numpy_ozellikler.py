import numpy as np

# ndim: Boyut sayısı
# shape: Boyut bilgisi
# size: Toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size = 10) # başlayacağı yeri yazmazsan 0 dan başlar
print(a.ndim)                       # 1 boyutlu
print("----------------------------------------")
print(a.shape)                      # 1 boyutlu 10 eleman
print("----------------------------------------")
print(a.size)                       # 10 eleman sayısı
print("----------------------------------------")
print(a.dtype)                      # int32 tipli
print("----------------------------------------")

## İKİ BOYUTLU ARRAY ##

b = np.random.randint(10, size = (3,5))
print(b)
print("----------------------------------------")
print(b.ndim,'\n',b.shape,'\n',b.size,'\n',b.dtype) # 2 boyutlu, (3,5) boyut, 15 eleman saıyıs
print("----------------------------------------")

######### RESHAPE ############

print(np.arange(1,10))
print("------------------------------------")
print(np.arange(1,10).reshape((3,3))) # (3,3) e dönüştürüldü.
print("------------------------------------")
print(np.arange(1,10).reshape((1,9))) # (1,9) a dönüştürüldü.
print("------------------------------------")












