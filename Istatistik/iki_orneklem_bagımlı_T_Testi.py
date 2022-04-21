import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
# Bir gruba eğitim veriliyor, öncesi ve sonrası karşılaştırılıyor.

oncesi = pd.DataFrame([123,119,119,116,123,123,121,120,117,118,121,121,123,119,
            121,118,124,121,125,115,115,119,118,121,117,117,120,120,
            121,117,118,117,123,118,124,121,115,118,125,115])

sonrasi = pd.DataFrame([118,127,122,132,129,123,129,132,128,130,128,138,140,130,
             134,134,124,140,134,129,129,138,134,124,122,126,133,127,
             130,130,130,132,117,130,125,129,133,120,127,123])
            
# BIRINCI TİP VERİ SETİ
ayrık = pd.concat([oncesi, sonrasi], axis = 1)
ayrık.columns = ["Oncesi", "Sonrası"]
print("'AYRIK' Veri Seti: \n\n ", ayrık.head(), "\n\n")

# IKINCI TİP VERİ SETİ
## ONCESİ FLAG/TAG'ı Oluşturma
GRUP_ONCESI = np.arange(len(oncesi))
GRUP_ONCESI = pd.DataFrame(GRUP_ONCESI)
GRUP_ONCESI[:] = "ONCESI"
## FLAG ve Oncesi değerlerini bir araya getirme
A = pd.concat([oncesi, GRUP_ONCESI], axis= 1)
## SONRASI FLAG/TAG'ını Oluşturma
GRUP_SONRASI = np.arange(len(sonrasi))
GRUP_SONRASI = pd.DataFrame(GRUP_SONRASI)
GRUP_SONRASI[:] = 'SONRASI'
## FLAG ve Sonrasını birleştirme
B = pd.concat([sonrasi,GRUP_SONRASI], axis= 1)

BIRLIKTE = pd.concat([A, B])
BIRLIKTE.columns = ["PERFORMANS", "ONCESI_SONRASI"]
print("'BIRLIKTE' Veri Seti: \n\n", BIRLIKTE.head(), "\n\n")

# VERİYİ TAMAMLADIK. Normalde bu halde csv gibi bir modda gelecek.
# Şimdi işimize başlayalım.

#### BOX PLOT ile Farka Bakalım
import seaborn as sns
sns.boxplot(x = "ONCESI_SONRASI", y = "PERFORMANS", data = BIRLIKTE)
#plt.show()   # Baya fark görünüyor ama test yapmamız lazım

# TEST YAPABİLMEK İÇİN VARSAYIM KONTROLÜ YAPACAGIZ.

#******************# NORMALLİK VARSAYIMI #******************#
from scipy.stats import shapiro
print(shapiro(ayrık.Oncesi))  # ShapiroResult(statistic=0.9543654918670654, pvalue=0.10722342133522034)
# H0 ı reddedemiyoruz. yani normal dağılmaktadır.
print(shapiro(ayrık.Sonrası))  # ShapiroResult(statistic=0.9780087471008301, pvalue=0.6159457564353943)
# H0 ı reddedemiyoruz. yani normal dağılmaktadır.

#******************# HOMOJENLİK VARSAYIMI #******************#
print(stats.levene(ayrık.Oncesi, ayrık.Sonrası)) # LeveneResult(statistic=8.31303288672351, pvalue=0.0050844511807370246)
# pvalue< 0.05  H0 reddedildi.
# 1. seçenek: Veri setinde bazı aykırılıklar varsa veri seti tekrardan düzenlenebilir.
# 2. seçenek: Bir miktar göz ardı edilebilir. şuan göz ardı edip öyle devam edeceğiz.

################    PARAMETRİK     #################

# TEST AŞAMASINA GEÇİYORUZ.
test_istatistiği, pvalue = stats.ttest_rel(ayrık.Oncesi, ayrık.Sonrası)
print('Test İstatistiği = %.4f, p-değeri =%.4f' % (test_istatistiği, pvalue))
# pvalue < 0.05 H0 REDDEDİLDİ
# SONUÇ YORUM
# H0 eğitim öncesi ve sonrası aynı diyordu ve bu reddedildi
## yani öncesi ve sonrası arasında fark vardır. Eğitim iyi etki etmiştir.


################    NON- PARAMETRİK     #################
# Normallik yoksa yani non parametrik ise wilcoxon testi yapıyoruz.

stats.wilcoxon(ayrık.Oncesi, ayrık.Sonrası)
test_istatistiği, pvalue = stats.wilcoxon(ayrık["Oncesi"],ayrık["Sonrası"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistiği, pvalue))
## yani öncesi ve sonrası arasında fark vardır. Eğitim iyi etki etmiştir.