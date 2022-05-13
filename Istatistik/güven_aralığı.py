import numpy as np

fiyatlar = np.random.randint(10,110,1000)
print(fiyatlar.mean())

import statsmodels.stats.api as sms

print(sms.DescrStatsW(fiyatlar).tconfint_mean()) # Güven aralığını bulmamıza yardımcı olur. Kabul edilen Güven aralığı %95'tir. 
# 56 ile 60 diye iki sonuç çıkarıyor, müşterilerin %95 güvenirlilikle 56-60 arası ödeme yaparak bu ürünü satın almak isteyeceğini söylüyor.

def yazdir(metin):
    print(metin, "program öğrenilecek")

yazdir("o")
