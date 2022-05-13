# Binom Dağılımı
# Bağımsız n deneme sonucu k başarılı olma olasılığı ile ilgilendiğimizde kullanacağız.
# Madeni para 4 kere atılıyor. 2 kere yazı gelme olasılığı nedir ?


# Reklam Harcaması Optimizasyonu
# Bir mecrada reklam verilecek. Dağılım ve reklama tıklama olasılığı 0.01, 
# Reklamı 100 kişi gördüğünde 1,5,10 tıklanma olasılığı nedir ?
from scipy.stats import binom

p = 0.01
n = 100
rv = binom(n, p)
print(rv.pmf(1))
print(rv.pmf(5))
print(rv.pmf(10)) 
