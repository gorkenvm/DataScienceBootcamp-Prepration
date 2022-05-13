# Olasılıkta 0 ve 1 aralığını anlatır.
# Başarılı-başarısız, Olumlu-olumusuz gibi iki sonuçlu olaylar ile ilgilenildiğinde kullanılır.
# Kategorik verilerde uygulanır

# Beklenen değer(mean) merkezi eğilimi ifade ediyor ve Varyans ise yayılımı ifade ediyor.

from scipy.stats import bernoulli
# Tura olma olasılığı 0.6, Yazı 0.4
p = 0.6
rv = bernoulli(p)
print(rv.pmf(k= 0)) # k = 0 yazı olma ihtimali, k = 1 tura olma olasığını hesaplar
