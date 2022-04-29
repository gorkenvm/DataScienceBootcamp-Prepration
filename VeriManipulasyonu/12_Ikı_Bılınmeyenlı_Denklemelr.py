import numpy as np
# PROBLEM
# 5*x0+x1 = 12
# x0+3x1=  10

a = np.array([[5,1], [1,3]])    # katsayıları ifade ettik
b = np.array([12,10])           # denklemin diğer tarafını ifade ettik

x = np.linalg.solve(a,b)    # linearalgebra methoduyla, a ve b arasındaki ilişkiyi çözdürüyoruz.
print(x)                    # x0 = 1.85714286, x1 = 2.71428571
