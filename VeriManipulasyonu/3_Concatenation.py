import numpy as np

x = np.array([1,2,3])
y = np.array([4,5,6])

print(np.concatenate([x,y])) # x y  birleştirildi
print("------------------------------")
z = np.array([7,8,9])
print(np.concatenate([x,y,z])) #x y z birleştirildi.
print("------------------------------")
###İKİ BOYUT ###
a = np.array([[1,2,3],[4,5,6]])
print(a)
print("------------------------------")
print(np.concatenate([a,a]))        # axis default = 0, alt alta birleştir
print("------------------------------")
print(np.concatenate([a,a], axis=1)) # axis =1 yan yana birleştir
