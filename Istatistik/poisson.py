import imp
from scipy.stats import poisson
# Lambda is 0.1 dir. 
# Bir siteye veri girilecek, 0, 3, 5 hata olma olasılıklarını hesapla.
lambda_ = 0.1
rv = poisson(mu = lambda_)

print(rv.pmf(0))  # Hiç hata olmama durumu 
print(rv.pmf(3))  # 3 hata olma
print(rv.pmf(5))  # 5 hata olma