# Websitesinde geçirilen sürenin 170 sn'ye olduğu hipotezi var.
# 50 valueli bir örneklem çekiyoruz.
from os import stat
import numpy as np

olcumler = np.array([17, 160, 234, 149, 145, 107, 197, 75, 201, 225, 211, 119, 
              157, 145, 127, 244, 163, 114, 145,  65, 112, 185, 202, 146,
              203, 224, 203, 114, 188, 156, 187, 154, 177, 95, 165, 50, 110, 
              216, 138, 151, 166, 135, 155, 84, 251, 173, 131, 207, 121, 120])

import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

print(stats.describe(olcumler)) # describe ile örneklmein ist. sonuçlarını kotnrol ediyoruz.
# Mean 154.38 çıktı. 
# T testi yapabilmek için Normallik şartı aranıyor.
#### VARSAYIMLAR

# Normallik Testi
df = pd.DataFrame(olcumler).plot.hist()
#plt.show()                                 # Normal dağılım gibi görünüyor.

#qqplot
import pylab
stats.probplot(olcumler, dist="norm", plot=pylab)
#pylab.show()                               # mavi noktalar kırmızı çizgi yakınında normal dagılım gibi görünüyor.

#Shapiro - Wilks Testi  ****** Önemli
## H0: Örnek dağılımı ile teorik dağılım arasında ist. ol. anl. bir fark. yoktur.
## H1: ..... fark. vardır.
# P value ye bakılır ve daha kesin sonuç elde edilir.
from scipy.stats import shapiro
print(shapiro(olcumler))   # ShapiroResult(statistic=0.9853105545043945, pvalue=0.7848747968673706)
# P value 0.05 ten küçük ise Ho ı red ediyoruz.
# p value 0.05 ten büyük yani 0.78'dir.
# Bu durumda parametrik bir örneklem testi olan T Testini uygulayabiliriz.
print(stats.ttest_1samp(olcumler, popmean= 170))  #Ttest_1sampResult(statistic=-2.1753117985877966, pvalue=0.034460415195071446)
# H0: Web sitemizde geçirilen ort. süre 170tir.
# H1: ..... değildir.
### p-value 0.05 ten küçük olduğu için Ho reddedilir. Yani 170 değildir.