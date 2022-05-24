# Amaç : Hata karaler toplamını minimize eden katsayıları bu katsayılara bir ceza uygulayarak bulmaktır.
# ElasticNet L1(Lasso) ve L2(Ridge) yaklaşımlarını birleştirir.
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV,ElasticNetCV

############################## VERİ SETİ ##############################
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Hitters.csv")
df = df.dropna() # Eksik değerleri uçurduk, konu kapsamında olmadığından dolayı
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])   # Kategorik değişkenleri one-hot-encoding yaklaşımı ile dummy'e çevirdik
y = df["Salary"]        # Bağımlı değişkenimiz.
X_ = df.drop(['Salary','League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
############################## MODEL KURMA ##############################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)
enet_model = ElasticNet().fit(X_train, y_train)
print(enet_model.coef_)
print(enet_model.intercept_)
############################## TAHMİN ##############################
print(enet_model.predict(X_train)[:10])
print(enet_model.predict(X_test)[:10])
y_pred = enet_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # ilkel test hatamız
print(r2_score(y_test, y_pred))                     # r2 skorumuz
############################## MODEL TUNNİNG ##############################
enet_cv_model = ElasticNetCV(cv = 10).fit(X_train, y_train) # lambdaları elasticnet'in bulmasını bekledik bu değerlere bakalım.
print(enet_cv_model.alpha_) # 5230 gibi bir alpha değeri buldu
print(enet_cv_model.intercept_) # -38.519 gibi bir sabit
print(enet_cv_model.coef_) # ve sıfıra yakın katsayılar buldu
# Ridge tarzı cezalandırma ve Lasso tarzı değişken seçimi yapıyor, dolayısıyla ikisinin özelliğini birleştiriyor.
#### FİNAL MODEL ####
enet_tuned = ElasticNet(alpha= enet_cv_model.alpha_).fit(X_train, y_train) # tuned edilmiş modelimizi fit ettik.
# şimdi final modeli kullanarak test hatamızı hesaplayalım
y_pred = enet_tuned.predict(X_test) # test bağımsızı ile test bağımlıyı tahmin ettik
print(np.sqrt(mean_squared_error(y_test, y_pred))) # gerçek test y ile, test x kullanılarak bulunan y nin test hatası
# final test hata sonucumuz budur 394.152,
# En sonunda farklı modeller kullanılarak bulunan final test hatalarını karşılaştırarak hangi modeli kullanacağımıza karar vermemiz gerekiyor.

# ÖNEMLİ ************
# alpha'yı elasticnet'ten istedik, istersek alphayı kendimizde alphas= parametresi ile oluşturulabiliriz. Lasso ve Ridgede yapmıştık.
# elasticnet( l1_ratio = 0.5 ) ön tanımlıdır. 0 olduğunda L2 cezalandırması, 1 olduğunda L1 cezalandırması yapılır.
# cezalandırmaların bir birine göre görecesini ifade eder. biz eşit olsun istediğimiz için 0.5 bıraktık.
# l1_ratio için de bir liste verilebilir. bir birleri arasındaki optimum sonuç ne zaman geldiği gözlemlenebilir.
# her bir l1_ratio değeri için farklı lambdalar kullanılmış olacaktır ve modele etki edeceklerdir.







