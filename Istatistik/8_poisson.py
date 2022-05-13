# Poisson : Belirli bir zaman aralığında belirli bir alanda nadiren rastlanan olayların olayların olasılıklarını hesaplamak için kullanılır
# Nadiren nedir ? 10.000 kelimeli kitapta hata sayısı, KK işlemlerinde sahtekarlık olayı, Rötara düşen uçuş sefer sayısı. 
# n gözlem sayısı, p olasılık. nadir kabul edilmesi için n > 50, n*p < 5 olmalıdır. n*p Lambdadır.
# Formülde Lambda var değiştirebiliyoruz, Lambda ortalama ve varansı ifade eder, Lambda artarsa yayılım artar.
# Kesikli bir rastsal dağılımdır.
# Binom dağılımının özel bir halidir.


# İş Uygulaması
# İlan giriş hata olasılıklarının hesaplanması
import imp
from scipy.stats import poisson
# Lambda is 0.1 dir. 
# Bir siteye veri girilecek, 0, 3, 5 hata olma olasılıklarını hesapla.
lambda_ = 0.1
rv = poisson(mu = lambda_)

print(rv.pmf(0))  # Hiç hata olmama durumu 
print(rv.pmf(3))  # 3 hata olma
print(rv.pmf(5))  # 5 hata olma
