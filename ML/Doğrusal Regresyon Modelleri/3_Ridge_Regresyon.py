
####################### TEORİ #################################3

# Amaç; hata karaler toplamını minimize eden katsayıları, bu hatsayılara bir ceza uygulayarak bulmaktır.
# Ceza uygulamak nedir ? formülü klasik regresyonla aynı buna ek olarak landa * Toplam (j=1, p) Bj kare  ekleniyor.
# lambda ya ayar parametresi diyoruz. sonraki kısıma ise ceza terimi diyoruz. Amaç fonksiyondaki Beta parametrelerini en ideale getirmektir. Lambda kullanıcı tarafından verilir
##### ÖZELLİKLERİ
# Aşırı öğrenmeye karşı dirençlidir.
# Yanlıdır fakat varyansı düşüktür. ( Bazen yanlı modelleri daha çok tercih ederiz )
# Çok fazla parametre olduğunda EKK(enküçükkareler)'ya göre yani klasik regresyona göre daha iyidir.
# Çok boyutluluk lanetine karşı çözüm sunar. Lanet nedir ? değişken sayısı gözlem sayısından büyük olduğunda.
# Çoklu doğrusal bağlantı problemi olduğunda etkilidir. Ç.D.B.P nedir? bağımsız değişkenler arasında yüksek korelasyon olmasıdır. Yani bir değişkenin taşıdığı bilginin aynısını neredeyse bir başka değişkende taşıyor.
## Bunu bulmak için korelasyona bakılabilir. iki değişken arasında %90 gibi bir korelasyon varsa bu bağlantı problemine sebep olacaktır.
# ** Tüm değişkenler ile model kurar, ilgisiz değişkenleri modelden çıkarmaz, katsayılarını sıfıra yaklaştırır.
# lambda kritik roldedir. iki terimin(formüldeki) göreceli etkilerini kontrol etmeyi sağlar
# lambda için iyi bir değer bulunması önemlidir. Bunun için CV yöntemi kullanılır.
# Lambda'nın 0 olduğu yer bildiğimiz klasik regresyondur.

### LAMBDA Parametresinin Belirlenmesi
# Öyle bir lambda bulmalıyız ki MSE'yi sıfır yapsın
# Lambda için belirli değerleri içeren bir küme seçilir ve her biri için CV test hatası hesaplanır.
# En küçük CV'ı veren lambda ayar parametresi olarak seçilir.
# Son olarka seçilen lambda ile model yeniden tüm gözlemlere fit edilir.

####################### UYGULAMA #################################
import numpy as np
import pandas as pd
pd.options.display.width = 0
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

############################## VERİ ##############################
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Hitters.csv")
df = df.dropna() # Eksik değerleri uçurduk, konu kapsamında olmadığından dolayı
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])   # Kategorik değişkenleri one-hot-encoding yaklaşımı ile dummy'e çevirdik
y = df["Salary"]        # Bağımlı değişkenimiz.
X_ = df.drop(['Salary','League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
# Özetle: Kategorikleri dummy'e çevirdik, dummy dışındakileri uçurup X_ dedik, X_ ile dummyleri birleştirdik.
# Yani y bağımlı değişkenimiz, X bağımsız değişkenlerimiz oldu.

############################## MODEL KURMA ##############################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)
# Soru: Ben bir oyuncu olacağım ama ne kadar maaş ödemem lazım ?
ridge_model = Ridge(alpha=0.1).fit(X_train, y_train)    # Model kuruldu. alpha lambda'dır.
print(ridge_model.coef_)    # Katsayıları gözlemledik, alpha değiştikçe katsayı değişecek.
# amacımız olan hata oranını en düşük yapmak için en uygun lambda katsayısını bulmalıyız.
print(ridge_model.intercept_)
lambdalar = 10**np.linspace(10,-2,100)*0.5
print(lambdalar)    # oluşturduğumuz lambdalara karşı coefler nasıl değişecek onu gözlemleyeceğiz.

ridge_model = Ridge()
katsayilar = []
for i in lambdalar:
    ridge_model.set_params(alpha = i)   # set_params = parametre olarak set et demek
    ridge_model.fit(X_train, y_train)   # fit et
    katsayilar.append(ridge_model.coef_)    # katsayilar'a modelin coef'ini ekle.
print(katsayilar[:2]) # Çok fazla olduğu için sadece ilk 2 tanesini yazdırıyoruz.
# Şimdi görselleştirip farka bakacağız.
ax = plt.gca()
ax.plot(lambdalar, katsayilar)
ax.set_xscale("log")    # ölçeklendirme yaptırdık.
#plt.show()              # Çok güzel bir graph geldi, görüldüğü gibi lambda artınca değişkenlerin etkilerini sıfıra yaklaştırıyor.

############################## TAHMİN ##############################
y_pred = ridge_model.predict(X_train)   # Tahmin edilen değerler
print(y_pred[:10])
print(y_train[:10])                     # Gerçek değerler
# Elimizde hem train hem de test hatası olacak.
# train hatasına bakalım. Modeli train'den kurdugumuz için bunun parametrelerini iyileştireceğiz.

# Train Hatası
RMSE = np.sqrt(mean_squared_error(y_train, y_pred))
print("RMSE:", RMSE) # Train setine ilişkin RMSE, valide edilmemiş RMSE

# CV Train RMSE
cv_RMSE = np.sqrt(np.mean(-cross_val_score(ridge_model, X_train, y_train, cv= 10, scoring= "neg_mean_squared_error")))
print("cvRMSE:", cv_RMSE)  # Valide edilmiş hata, bu rakam valide edilmemiş olana göre daha doğrudur.

# Test Hatası
y_pred = ridge_model.predict(X_test)    # y'nin tahminleri
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSEtest:",RMSE)

#*** Bütün modelleri bir birleri ile kıyaslamak için TEST HATALARINI KIYASLAYACAĞIZ:
#*** EĞİTİM HATALARINI DA modelleri tunning etmek için en iyi hale getirmek için, parametreleri ayarlamada kullanacağız.
#*** Yani CV 'yi sadece train setlerinde kullanacağız.

############################## MODEL TUNNİNG ##############################
# lambda parametresini optimize ederek model tunning yapmış olacağız.
#  Neden Model Tunning ?
ridge_model = Ridge(100).fit(X_train, y_train)  # Model oluşturuldu Eğitim verileriyle
y_pred = ridge_model.predict(X_test)            # test verisi ile yşapka tahmin edildi
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))     # gerçek veri ile yşapka arasındaki farktan hatamızı bulduk
# Ridge içine yazdığımız lambda 100 değeri ne olmalı ki model bize en iyi hata sonucunu versin bunu yapmaktır tunning etmek
lambdalar1 = np.random.randint(0,1000,100)  # 0'dan 1000'e kadar 100 tane rastgele değer ürettik, hepsini deneyerek en iyi almbdayı bulmaya çalışacağız.
lambdalar2 = 10**np.linspace(10, -2, 100)*0.5 #
# ridgecv tüm alphaları deneyecek
# Lambdalar2
ridgecv = RidgeCV(alphas= lambdalar2, scoring="neg_mean_squared_error", cv = 10, normalize= True) # istersek rkare, istersek hatakareoranını yazabiliriz
ridgecv.fit(X_train, y_train)   # fit edince gerekli hesaplamaları yapıyor.
print("ridgecvalpha:",ridgecv.alpha_)   # optimum lambda 0.7599555414764666 dır deniliyor.
# FİNAL MODELİ
ridge_tuned = Ridge(alpha= ridgecv.alpha_).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # RMSE 356.85830472715173 çıktı.
#### Aynısını Lambdalar1 için yapalım
ridgecv = RidgeCV(alphas= lambdalar1, scoring="neg_mean_squared_error", cv = 10, normalize= True)
ridgecv.fit(X_train, y_train)
print("ridgecvalpha2:",ridgecv.alpha_)  # alpha 7 çıktı,
# FİNAL MODEL Lambdalar1 için
ridge_tuned = Ridge(alpha= ridgecv.alpha_).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # RMSE: 356.330306002515 çıktı.



