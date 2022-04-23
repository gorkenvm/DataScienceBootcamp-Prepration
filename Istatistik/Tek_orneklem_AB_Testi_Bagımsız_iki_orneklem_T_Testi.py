# Gelir arttırmak için ürün tavsiye eden ML sistemi kurulup entegre ediliyor.
# Bir gruba eski sistem bir gruba yeni sistem gösteriliyor. A ve B sistemleri diyelim.
# Bu sistemleri karşılaştırıp testlerini yapacağız.

# Veri bize hazır gelecek ama veriyi düzenleyip gruplamayı da öğrenelim.

from pstats import Stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


####### VERI TIPI I
A = pd.DataFrame([30,27,21,27,29,30,20,20,27,32,35,22,24,23,25,27,23,27,23,
        25,21,18,24,26,33,26,27,28,19,25])

B = pd.DataFrame([37,39,31,31,34,38,30,36,29,28,38,28,37,37,30,32,31,31,27,
        32,33,33,33,31,32,33,26,32,33,29])

A_B = pd.concat([A, B], axis = 1) # Verileri birleştirip columlara yerleştirdik.
A_B.columns = ["A", "B"]          # column isimlerini A ve B yaptık.
print(A_B.head())


####### VERI TIPI II      Bu biraz amele yöntemi ama bazen gerek olabildiği için öğrenelim.
# A ve A'nın Grubu
Grup_A = np.arange(len(A))       # A veri sayısı uzunlugunda bir seri oluşturuldu.
Grup_A = pd.DataFrame(Grup_A)    # bu seri dataframe'e aktarıldı.
Grup_A[:] = "A"                  # grup ismi de A yapıldı.
A = pd.concat([A, Grup_A], axis = 1) # Column olarak birleştirildi.

# B ve B'nın Grubu
Grup_B = np.arange(len(B))
Grup_B = pd.DataFrame(Grup_B)
Grup_B[:] = "B"
B = pd.concat([B, Grup_B], axis = 1)


# Tum veri
AB = pd.concat([A, B]) # A ve B birleştirildi.
AB.columns = ["gelir", "GRUP"]
print(AB.head())
print(AB.tail())

# Box plot ile arada fark varmı bakalım.
sns.boxplot(x = "GRUP", y = "gelir", data= AB)
#plt.show()

# Evet Fark var ve B daha yüksek görünüyor.
# Şimdi Test etmemiz gerekir.

# Test Yapabilmek için VARSAYIM KONTROLÜ yapacağız.
# NORMALLİK VARSAYIMI
from scipy.stats import shapiro

print(shapiro(A_B.A)) # ShapiroResult(statistic=0.9789242148399353, pvalue=0.7962799668312073)
# p value > 0.05 H0 red edilmedi
print(shapiro(A_B.B)) # ShapiroResult(statistic=0.9561261534690857, pvalue=0.2458445429801941)
# p value > 0.05 H0 red edilmedi
# Varsayım kontrolü başarıyla tamamlandı.

# VARYANS HOMOJENLİK TESTİ
#H0: Varyanslar Homojendir
#H1: Varyanslar Homojen değildir.
# levene testini kullanacağız.
print(stats.levene(A_B.A, A_B.B)) #LeveneResult(statistic=1.1101802757158004, pvalue=0.2964124900636569)
# H0 red edilemedi yani Varyans homojendir.

############ HİPOTEZ TESTİ  ############
#################################PARAMETRİK##################################
# Yukarıdaki VARSAYIM şartlarını sağlandığı için;
# PARAMETRİKTİR DİYOR VE ttest_ind testi yapıyoruz. **************
print(stats.ttest_ind(A_B["A"], A_B["B"], equal_var = True)) # Ttest_indResult(statistic=-7.028690967745927, pvalue=2.6233215605475075e-09)
# p value < 0.05 H0 reddildi. Yani modeller arasında fark vardır.

### Daha iyi bir sonuç gösterimi için aşağıdaki gibi yapabiliriz.

test_istatistigi, pvalue = stats.ttest_ind(A_B["A"], A_B["B"], equal_var = True) # sonucu iki farklı parametreye atadık.
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# Sonuç böyle görünür: ---> Test İstatistiği = -7.0287, p-değeri = 0.0000 <-----

###############################NONPARAMETRİK##################################
# Yukarıdaki VARSAYIM şartlarını sağlandığında;
# NONPARAMETRİKTİR DİYOR VE mannwhitneyu testi yapıyoruz. **************

stats.mannwhitneyu(A_B["A"], A_B["B"]) # burayı print etmedim aşağıyı ile aynı şey.
# Daha iyi görsel için aşağıyı yapıyoruz.
test_istatistigi, pvalue = stats.mannwhitneyu(A_B["A"], A_B["B"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

#Done