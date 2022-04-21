import numpy as np

fiyatlar = np.random.randint(10,110,1000)
print(fiyatlar.mean())

import statsmodels.stats.api as sms

print(sms.DescrStatsW(fiyatlar).tconfint_mean()) # Güven aralığını bulmamıza yardımcı olur. Kabul edilen Güven aralığı %95'tir.

def yazdir(metin):
    print(metin, "program öğrenilecek")

yazdir("o")