# Haber sitesi A, B, C şeklinde 3 farklı strateji izlenecek.
# Buna göre websitesinde geçirilen süreler kaydedilecek.
import abc
from scipy.stats import shapiro
import scipy.stats as stats
import pandas as pd

A = pd.DataFrame([28,33,30,29,28,29,27,31,30,32,28,33,25,29,27,31,31,30,31,34,30,32,31,34,28,32,31,28,33,29])

B = pd.DataFrame([31,32,30,30,33,32,34,27,36,30,31,30,38,29,30,34,34,31,35,35,33,30,28,29,26,37,31,28,34,33])

C = pd.DataFrame([40,33,38,41,42,43,38,35,39,39,36,34,35,40,38,36,39,36,33,35,38,35,40,40,39,38,38,43,40,42])

dfs = [A, B, C]
ABC = pd.concat(dfs, axis = 1)
ABC.columns = ["GRUP_A", "GRUP_B", "GRUP_C"]
print(ABC)


############ VARSAYIM KONTROLÜ ####################
# NORMALLİK
from scipy.stats import shapiro
print(shapiro(ABC["GRUP_A"])) # ShapiroResult(statistic=0.9697431921958923, pvalue=0.5321715474128723)
print(shapiro(ABC["GRUP_B"])) # ShapiroResult(statistic=0.9789854884147644, pvalue=0.7979801297187805)
print(shapiro(ABC["GRUP_C"])) # ShapiroResult(statistic=0.9579201340675354, pvalue=0.273820161819458)
# pvalue > 0.05 her 3 grup içinde. H0 reddedilemedi.

# HOMOJENLİK
print(stats.levene(ABC["GRUP_A"], ABC["GRUP_B"], ABC["GRUP_C"])) # LeveneResult(statistic=1.0267403645055275, pvalue=0.3624711011741707)
# p value > 0.05 
# Normallik ve homojen varsayımları sağlanmıştır.

############ HİPOTEZ TESTİ ####################
##### PARAMETRİK ########
from scipy.stats import f_oneway

print(f_oneway(ABC["GRUP_A"], ABC["GRUP_B"], ABC["GRUP_C"])) # F_onewayResult(statistic=74.69278140730431, pvalue=1.307905074681148e-19
# pvalue < 0.05 H0 Reddedildi.
# Gruplar arasında ist. olarak anlamlı farklılık vardır.
# Hangi grup seçilmeliye bakıyoruz.

print(ABC.describe().T) 
# Sonuca göre C daha yüksek o seçilmelidir.

##### NON-PARAMETRİK ########
#Varsayımlar sağlanmamış olsaydı Non parametrik olacaktı.
from scipy.stats import kruskal
print(kruskal(ABC["GRUP_A"], ABC["GRUP_B"], ABC["GRUP_C"])) # KruskalResult(statistic=54.19819735523783, pvalue=1.7022015426175926e-12)
# Yine pvalue < 0.05 oldugundan H0 reddedilecekti.






