# AMAÇ: hata kareler toplamını minimize eden katsayıları bu katsayılara bir ceza uygulayarak bulmaktır.
# Ridge ile amaç aynı. Lasso için L1, Ridge için L2 yöntemi de denmektedir.
# Formülde tek fark: Ridge ceza teriminin karesini alırken, Lasso Mutlak değerini alıyor.
# AVANTAJ DEZAVANTAJLAR
# Ridge'nin tüm değişkenleri modelde bırakma dezavantajını gidermek için önerilmiştir.
# Lasso'da katsayılar sıfıra yaklaştırır
# Fakat L1 normu lambda yeteri kadar büyük olduğunda bazı katsayıları sıfır yapar. Böylece değişken seçimi yapılmış olur.
# Lambdanın doğru seçilmesi çok önemlidir, burada da CV kullanılır.
# Ridge ve Lasso yöntemleri bir birinden üstün değildir.

##### LAMBDA'nın belirlenmesi
# Lambda'nın sıfır olduğu yer EKK'dır. HKT(hatakaralertoplamı)'yi minimum yapan lambda'yı arıyoruz
# Lambda için belirli değerleri içeren bir küme seçilir ve her birisi için CV test hatası hesaplanır.
# En küçük CV'ı veren lambda ayar parametresi olarak seçilir.
# Son olarak seçilen bu lambda ile model yeniden tüm gözlemlere fit edilir.

####################### UYGULAMA #################################
import numpy as np
import pandas as pd
pd.options.display.width = 0
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score

############################## VERİ ##############################
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Hitters.csv")
df = df.dropna() # Eksik değerleri uçurduk, konu kapsamında olmadığından dolayı
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])   # Kategorik değişkenleri one-hot-encoding yaklaşımı ile dummy'e çevirdik
y = df["Salary"]        # Bağımlı değişkenimiz.
X_ = df.drop(['Salary','League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)

############################## MODEL KURMA ##############################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)
lasso_model = Lasso().fit(X_train, y_train)
print("Sabit:",lasso_model.intercept_, "\n","katsayılar:","\n",lasso_model.coef_)   # Sabit ve katsayıları aldık.
# Farklı lambda değerlerine karşılık katsayılar
lasso = Lasso()
coefs = []
alphas = 10**np.linspace(10, -2, 100)*0.5   # rastgele alphalar oluşturduk
for a in alphas:
    lasso.set_params(alpha = a )    # alpha'yı ayarla
    lasso.fit(X_train, y_train)     # modeli kur
    coefs.append(lasso.coef_)       # coef'i al ve coefs listesine ekle
# katsayıları görselleştirme
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
plt.show()              # Burada lambda büyüdükçe katsayıların sıfıra yaklaştığını gözlemlemek istedik.

############################## TAHMİN ##############################

print(lasso_model.predict(X_train)[:5])     # ilk 5 eğitim verisiyle tahmin ettik
print(lasso_model.predict(X_test)[:5])      # ilk 5 test verisiyle tahmin ettik

## TEST HATASI
y_pred = lasso_model.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print(RMSE)         # hata: 356.0975884554034, lasso ridgeye göre daha düşük hata verdi.
r2_score(y_test, y_pred)    # Mutlaka incelenmesi gereken score'lardan biridir. bağımsız değişkenlerce bağımlı değişkenin açıklanma yüzdesidir

############################## MODEL TUNNİNG ##############################

lasso_cv_model = LassoCV(cv= 10, max_iter= 1000).fit(X_train, y_train)
print(lasso_cv_model.alpha_)    # alpha yani lambda olara 563.4670501833854 rakamını tavsiye etmiş.
# Tavsiye edilen alpha değerini kullanarak tuned edelim
lasso_tuned = Lasso().set_params(alpha = lasso_cv_model.alpha_).fit(X_train, y_train) # tavsiye edilen alphayı koyduk ve fit ettik
y_pred = lasso_tuned.predict(X_test)        # X_test verisi ile test verisindeki yşapkaları bulduk
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # RMSE: 373.5957225069795 bulduk.
##### Lambda listesi ile en uygununu bulmaya çalışalım
alphas = 10**np.linspace(10,-2,100)*0.5
lasso_cv_model = LassoCV(alphas= alphas, cv = 10, max_iter= 100000).fit(X_train, y_train)
print( lasso_cv_model.alpha_)   # liste içindeki optimum alpha değeri = 201.85086292982749 dedi
lasso_tuned = Lasso().set_params(alpha = lasso_cv_model.alpha_).fit(X_train, y_train) # tavsiye edilen alphayı koyduk ve fit ettik
y_pred = lasso_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # RMSE 363.6832708037447 çıktı. daha düşük.

# Katsayılarımızı gözlemlemek istersek
print(pd.Series(lasso_tuned.coef_, index=X_train.columns))  # Hangi değişkenin maaş'a etkisinin olmadığını, hangisinin ne kadar etkili olduğunu gözlemleyebiliriz.
















