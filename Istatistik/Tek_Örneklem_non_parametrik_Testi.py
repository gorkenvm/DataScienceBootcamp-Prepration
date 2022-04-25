# Kullanacağımız örnek Normal dağılım yani parametrik bu yüzden sonucu dikkate alma. Sadece nasıl yapıldığı gösteriliyor.
# Nonparametrik olunca bunu yapabiliyoruz.
from statsmodels.stats.descriptivestats import sign_test
from Tek_orneklem_T_Testi import olcumler                # bir daha yazmak yerine olcumler'i Tek_orneklem_T_Testi script'inden import ettik
print(sign_test(olcumler, 170)) # Result (-7.0, 0.06490864707227217)
# p value 0.05 ten büyük olduğu için Ho'ı reddedemeyecektik.