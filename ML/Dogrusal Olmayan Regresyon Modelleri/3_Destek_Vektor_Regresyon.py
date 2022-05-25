# SVR : Güçlü ve esnek modelleme tekniklerindendir.
# Sınıflandırma ve regresyon için kullanılır
# Robust(dayanıklı) bir regresyon modelleme tekniğidir. Robust ile aykırı gözlemlere dayanıklı olduğunu ifade eder.
# Amaç : bir marjin aralığına maksimum noktayı en küçük hata ile alabilecek şekilde doğru ya da eğriyi belirlemektir.
# Eğriye paralel iki doğru oluşturulur. doğru ile uzaklığı epsilondur. oluşturulan doğru ile dışında kalan aykırı gözlemlerin uzaklığına kısi adı verilir
# Lasso ve Ridge deki gibi ceza terimleri vardır

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Hitters.csv")
df = df.dropna() # Eksik değerleri uçurduk, konu kapsamında olmadığından dolayı
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])   # Kategorik değişkenleri one-hot-encoding yaklaşımı ile dummy'e çevirdik
y = df["Salary"]        # Bağımlı değişkenimiz.
X_ = df.drop(['Salary','League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)

############################## MODEL ve TAHMİN ##############################
svr_model = SVR(kernel='linear').fit(X_train, y_train)
print(svr_model.get_params())   # Değiştirilebilecek parametreler ve ön tanımlarını inceleyebiliriz.
print(svr_model.predict(X_train)[:5])   # tahminlerin ilk 5i
print(svr_model.intercept_)             # B0 sabit katsayı
print(svr_model.coef_)                  # B1 katsayılar
### Test Hatası
y_pred = svr_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
# kernel linear kullandık. istersek rbf(radial bases function) de kullanabiliriz
############################## MODEL TUNİNG #########################
svr_model = SVR(kernel="linear")
svr_params = {"C": [0.1, 0.5, 1, 3]}    # rastgele değerler verdik ceza terimi için
#svr_cv_model = GridSearchCV(svr_model, svr_params, cv=5).fit(X_train, y_train)
svr_cv_model = GridSearchCV(svr_model, svr_params, cv=5, verbose=2, n_jobs= -1).fit(X_train, y_train)
# verbose = 2 işlemin ne kadar sürdüğü ve ne kadar işlem yapıldığı bilgisi
# n_jobs = -1 tüm işlemci gücünü kullan demek. bu yüzden bu parametreleri yazmak faydalı olacaktır.
print(svr_cv_model.best_params_)    # en iyi parametreyi 0.5 buldu
#### Final Model
svr_tuned = SVR(kernel="linear", C= 0.5).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))    # tuned edilmeden önce 370 idi tuned edilince 367ye düştü



















