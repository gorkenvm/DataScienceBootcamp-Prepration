import numpy as np

a  = np.random.randint(10, size = (5,5))
print(a)
print("--------------------------------------")
alt_a = a[0:3, 0:2]
print(alt_a)
print("--------------------------------------")
alt_a[0,0] = 999999         # Burada yaptığımız değişiklikler a'da da oldu.
alt_a[1,1] = 88888
print(alt_a)
print("--------------------------------------")
print(a)

m  = np.random.randint(10, size = (5,5))
alt_b = m[0:3, 0:2].copy()          # copy dediğimiz için m de bozulma olmadı.





