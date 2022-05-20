# Amaç: Bağımlı ve bağımsız değişkenler arasındaki ilişkiyi ifade eden doğrusal fonksiyonu bulmaktır.
# Hata Kareler Ortalamasını minimum yapmaya çalışacağız.
# Çoklu Doğrusal Regresyonda genelde 2 amaç vardır;
# 1- Bağımlı değişkeni etkilediği belirlenen bağımsız değişkenler vasıtasıyla, bağımlı değişkenin değerlerini tahmin etmek.
# 2- Bağımlı değişkeni etkilediği belirlenen bağımsız değişkenlerden, hangisinin yada hangilerinin bağımlı değişkeni daha çok etkilediğini tahmin etmek ve aralarındaki ilişkiyi tanımlamaya çalışmaktır.
# Yi = B0 + B1Xi1 + B2Xi2 + ..... +BpXip + Ei böyle bir fonksiyonumuz olacak. bunu çözüp E1 i minimum hale getirmeye çalışacağız.
# Paremetreleri bulmaya çalışacağız, Transpose kullanacağız, gözlem sayısı 1000 üzerine çıkınca bu yöntemi kullanamayacağız. Farklı yöntemler kullanacagız.
### Varsayımlar
# Neden-Sonuç ilişkisi kurabilmek için bu varsayımları sağlamak lazım
# - Hatalar normal dağılır
# - Hatalar birbirinden bağımsızdır ve aralarında otokorelasyon yoktur
# - Her bir gözlem için hata terimleri varyansları sabittir.
# - Değişkenler ile hata terimi arasında ilişki yoktur.
# - Bağımsız değişkenler arasında çoklu doğrusal ilişki problemi yoktur.

""""
Regresyon Modellerinin Avantajları ve Dezavantajları
- İyi anlaşılırsa diğer tüm ML ve DL konuları çok rahat kavranır
- Doğrusallık nedensellik yorumları yapılabilmesini sağlar, bu durum aksiyoner ve stratejik modelleme imkanı verir.
- Değişkenlerin etki düzeyleri ve anlamlılıkları değerlendirilebilir.
- Bağımlı değişkendeki değişkenliğin açıklanma başarısı ölçülebilir. rkare değeri ile
- Model anlamlılığı değerlendirilir
Negatif: - Varsayımları vardır. - Aykırı gözlemlere duyarlıdır.
"""
#################### MODEL #####################
import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Advertising.csv")
df = df.iloc[:,1:] # ilk columnda işimize yaramayacak sıra numaraları vardı sildik.
print(df.head())
X = df.drop('sales', axis = 1)
y = df[["sales"]]                 # Tek [] kullanırsak np array verir. İKİ [[]] kullanırsak df verir
print(y.head())
print(X.head())

# Statsmodels ile model kurmak
import statsmodels.api as sm
lm = sm.OLS(y, X)   # lm linearmodel kurduk OLS ile
model = lm.fit()    # modeli kurduk.
print(model.summary())      # Bu çıktı Çok değerli
# r-squared = bağımsız değişkenlerin bağımlı değişkenini açıklama değeri. 0.98 çok iyi bir rakam.
# Adj-R-squared = r-squared'in düzeltilmiş halidir, bunu dikkate alacağız.
# F-statistic = modelin anlamlılığını ifade etmek için kullanılır.
# Prob = pvalue'dir. 0.05'ten küçük oldugu için model anlamlıdır.

# coef = Kurulacak modeldeki bağımsız değişkenlerin katsayılarını ifade ediyor. B1 B2 B3 gibi
# std err = coef'lerin standard hatalarını verir
# t ve P>|t| = Elimizde coef'ler var ama bunlar anlamlı mıdır'ı söyler bize. t istatistiği ve pvalue değeri ile söyler.
# incelendiğinde pvalue 0.05'ten küçük oldugu için anlamlıdır.

# coef TV 0.05 ne demektir ? 1 birim TV harcaması 0.05 birim sales'e etki eder.


# Scikit learn ile model kurmak
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X, y)        # Model kuruldu
print(model.intercept_)     # Sabit
print(model.coef_)          # Katsayılar
# stats model ve sklearn ile kurduğumuz modellerdeki katsayılarda değişiklikler gözlemleyebiliyoruz.
# arkaplanda çalışan algoritmaların farklı olmasından kaynaklanmaktadır. Genelde sklearn kullanacağız.


#################### TAHMİN #####################
# Sales = 2.94 + TV*0.04 + radio*0.19 - newspaper*0.001
# Fonksiyonumuz hazır TV radio ve newspaper'a değer girerek sales'i tahmin edeceğiz.
# Örneğin// 30 birim TV, 10 birim radio, 40 birim gazete harcaması yaparsak sales ne olur ?
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T   # row şeklinde değilde column şeklinde olması için transpose aldık.
print(yeni_veri)
print(model.predict(yeni_veri))     # Bu birimleri harcarsak sales = 6.15562918 şeklinde tahminde bulunmuş olduk.
yeni_veri2 = [[300], [120], [400]]
yeni_veri2 = pd.DataFrame(yeni_veri2).T
print(model.predict(yeni_veri2))            # Veri dışındaki rakamları girerekte tahmin alabiliriz ki amaç budur.

#################### MODEL BAŞARI DEĞERLENDİRME #####################
from sklearn.metrics import mean_squared_error
# X bağımsız değişkenler
# y bağımlı değişken olan sales
print(model.predict(X)[:10])    # bağımsız değişkenleri kullanarak tahmin edilen yşapka değerleri
MSE = mean_squared_error(y, model.predict(X))
RMSE = np.sqrt(MSE)
print("MSE :", MSE)
print("RMSE :", RMSE)

#################### MODEL TUNİNG ( MODEL DOĞRULAMA ) #####################
# Burada daha çok model tuning değilde model doğrulama yapacağız.
# Sınama seti
from sklearn.model_selection import train_test_split
# Train ve test olarak %80'e %20 bölme işlemi yapıoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 99) # test_size = %80'e %20 ayır. random_state = Herseferinde farklı ayırma yapmaması için 99 giriyoruz.
print(X_train.head()) # train bağımsız veriler
print(y_train.head()) # train bağımlı veriler
print(X_test.head())  # test bağımsız veriler
print(y_test.head())  # test bağımlı veriler
# Eğitim seti üzerinden model kuruyoruz
lm = LinearRegression()
model = lm.fit(X_train, y_train)   # Eğitim setinin bağımlı ve bağımsız setini yazdık.
print(np.sqrt(mean_squared_error(y_train, model.predict(X_train)))) # Eğitim seti RMSE yi bulduk.
# eğitim hatası: 1.7236824822650751
print(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
# test hatası: 1.4312783138301641
# Eğitim ve test hatalarını bulduk.

### k-katlı cross validation
# Eğitim ve test'i %80'e %20 olarak ayırdık fakat hangi 80'e 20 olması gerektiğini bilmiyoruz.
# Bu yüzden k-katlı cv yaparak bir çok 80'e 20 deneyerek sonucu sağlamlaştırıyoruz.
from sklearn.model_selection import cross_val_score
print(cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")) # cv= kaç katlı, scoring= hangi skorlama; burada hata karaler ortalaması dedik
# [-2.1019073  -2.48953197 -3.09704214 -2.34694216 -3.68175761 -1.8691401 -3.18173007 -4.1927349  -2.17128376 -8.03821974]
#  Yukarıdaki gibi sonuç çıktı, ne yaptı ? eğitim setini 10 parçaya böldü, 9 tanesi ile model oluşturup 1 tanesi ile test etti ve hata buldu, sonra dışarıda bıraklan parça değiştirilerek 10 defa denendi.
# Hepsi - olduğu için en başa eksi koyalım. yöntemle alakalı hep eksi çıkıyor.
print(-(cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")))
# Bunların ortalaması eğitim ortalamasını verecek yani MSE, birde karekökünü alırsak RMSE'yi verecektir.
# cross validation edilmiş RMSE
print(np.sqrt(np.mean(-(cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")))))   # RMSE : 1.8212712522395245 bulduk.
# cv yapılmamış RMSE: 1.7236824822650751
# cv yapılmış   RMSE : 1.8212712522395245
# Neden cv yapıyoruz ? Sadece test ve train'in vereceği hatadan daha doğru bir hata sonucu verir.
# Bu modeli doğrulamaktır. Doğrulanmıştır.












