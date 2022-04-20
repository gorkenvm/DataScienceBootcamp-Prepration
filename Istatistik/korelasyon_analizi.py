# Değişkenler arasındaki ilişki yönü ve şiddeti hakkında bilgi sahibi oluruz.
# Tips veri seti ile çalışacağız.
# Bahşiş ve hesap tutarı arasında ki ilişkiyi inceleyeceğiz.
# total_bill: yemeğin fiyatı ( bahşiş ve vergi dahil)
# time: gündüz gece (0, 1)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro

tips = sns.load_dataset('tips')
df = tips.copy()
print(df.head())
# Toplam hesabın içinde bahşişte dahil oldugu ıcın bahşiş'i çıkarıyoruz.
df["total_bill"] = df["total_bill"] - df["tip"]
print(df.head())
# Veriyi görselleştirelim ki daha iyi gözlem yapabilelim.
df.plot.scatter("tip", "total_bill")
#plt.show()

####### VARSAYIM KONTROLÜ ##########
# NORMALLİK VARSAYIMI #
test_istatistiği, pvalue = shapiro(df["tip"])
print('Test istatistiği = %.4f, p-değeri = %.4f' % (test_istatistiği, pvalue))
test_istatistiği, pvalue = shapiro(df["total_bill"])
print('Test istatistiği = %.4f, p-değeri = %.4f' % (test_istatistiği, pvalue))
# pvalue < 0.05 H0 reddedildi. Yani aralarında anlamlı farklılık vardır.
# Her iki değişken içinde NORMALLİK Varsayımı Sağlanmadı NON PARAMETRİKTİR.

############## HİPOTEZ TESTİ #############
# PARAMETRİK # # Örneğimiz için NON PARAMETRİK yapmalıyız. Bunu öğrenmek için yapıyoruz.
# KORELASYON KATSAYISI
df["tip"].corr(df["total_bill"])  # corr: pearson katsayısını verir. Normal dağılımda kullanılır.
# 0.57  çıktı. Ne diyor bu rakamlar bize
# Değişkenler arasında pozitif bir ilişki var
# 0.5-0.6 arası yani orta şiddetli bir ilişki vardır.

# KORELASYON ANLAMLILIĞININ TESTİ
from scipy.stats.stats import pearsonr
test_istatistiği, pvalue = pearsonr(df["tip"], df["total_bill"])
print('Korelasyon katsayısı = %.4f, p-değeri = %.4f' % (test_istatistiği, pvalue))
# pvalue < 0.05 H0 = değişkenler arası anlamlı bir farklılık yoktur diyen H0 hipotezi reddedildi
# Yani değişkenler arasında anlamlı bir ilişki vardır.

# NON - PARAMETRİK
from scipy.stats import stats
test_istatistiği, pvalue = stats.spearmanr(df["tip"], df["total_bill"])
print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistiği, pvalue))
# pvalue < 0.05 H0 reddedildi yani pozitif orta şiddetli bir korelasyon vardır.

# ALTERNATİF OLARAK kendallta testi de uygulayabiliriz.
test_istatistiği, pvalue = stats.kendalltau(df["tip"], df["total_bill"])
print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistiği, pvalue))