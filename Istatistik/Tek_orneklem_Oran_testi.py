# Reklam 500 defa görülüyor 40 defa web sitesi tıklanıyor.
#40/500= 0.08 oluyor. 
# HO: p  = 0.125
# H1: p != 0.125  hipotezleri kuruluyor.

from statsmodels.stats.proportion import proportions_ztest

count = 40
nobs = 500
value = 0.125

print(proportions_ztest(count, nobs, value)) # (-3.7090151628513017, 0.0002080669689845979)

# p value 0.05 olduğu için H0 Red ediyoruz. %95 güvenirlilikle 0.125 doğru değildir.