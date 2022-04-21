# Arayüz tasarımında HEMEN-AL tuşu YEŞİL VE KIRMIZI 
# Hangisi olması gerektiğine karar vereceğiz.
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

basari_sayisi = np.array([300,250])
gozlem_sayiları = np.array([1000,1100])

print(proportions_ztest(count = basari_sayisi, nobs = gozlem_sayiları)) # (3.7857863233209255, 0.0001532232957772221)
# p value < 0.05 Oldugu için H0 reddedildi.
# H0 her iki buton aynıdır diyordu. Bunu reddettik aynı değildir.
