# Binom Dağılımı
from scipy.stats import binom

p = 0.01
n = 100
rv = binom(n, p)
print(rv.pmf(1))
print(rv.pmf(5))
print(rv.pmf(10)) 