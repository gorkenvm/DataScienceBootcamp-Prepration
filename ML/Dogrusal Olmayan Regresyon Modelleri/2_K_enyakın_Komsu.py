# Gözlemlerin birbirine olan benzerlikleri üzerinden tahmin yapılır.
# Sınıflandırma veya regresyon problemleri için kullanılır
# Parametrik olmayan bir öğrenme türüdür.
# Büyük veri setlerinde çok iyi değildir.
# Nasıl Hesaplanır; Öklid uzaklık yaklaşımı ile
# Basamaklar; 1- Komşu sayısı belirlenir. 2- Bilinmeyen nokta ile diğer tüm noktalar arasında uzaklıklar hesaplanır
# 3- Uzaklıkları sırala ve belirlenen k sayısına göre en yakın olan komşuları seç
# 4- Sınıflandırmada en yakın sınıf, regresyonda ortalama değeri tahmin değeri olarak ver.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

from warnings import filterwarnings # Uyarıların çıkmasını engeller.
filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Hitters.csv")
df = df.dropna() # Eksik değerleri uçurduk, konu kapsamında olmadığından dolayı
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])   # Kategorik değişkenleri one-hot-encoding yaklaşımı ile dummy'e çevirdik
y = df["Salary"]        # Bağımlı değişkenimiz.
X_ = df.drop(['Salary','League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)

############################## MODEL KURMA ve TAHMİN ##############################
knn_model = KNeighborsRegressor().fit(X_train, y_train)
print(knn_model.predict(X_test)[:5])    # test verisindeki bağımsız değişkenler ile bağımlı değişkenii tahmin etme
y_pred = knn_model.predict(X_test)  # tahmin edilenleri assign ettik
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # ilkel test hatası
############################## MODEL KURMA ve TAHMİN ##############################
# komşu sayısının değerlerini el yordamıyla farklı değerler yazarak gözlemleyelim.
RMSE = []
for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors= k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE.append(rmse)
    print("k=", k, "için RMSE değeri:", rmse)
# K 8 en düşük hata ortalamasını verdi. Bunu elle yaptık.

# GridSearchCV ile bunu fonksiyonel olarak yapabiliriz. Hiper parametreklerin değerlerini belirlemek için kullanılır
# olası tüm değerleri değerlendirir ve kıyaslar. Bundan sonra sıkça kullanacağız.
knn_params = {"n_neighbors": np.arange(1,30,1)}
knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn, knn_params, cv = 10).fit(X_train, y_train)
print(knn_cv_model.best_params_)    # Burada da fonksiyonel bir şekilde bulduk.
###### Final Model ######
knn_tuned = KNeighborsRegressor(n_neighbors= knn_cv_model.best_params_["n_neighbors"]).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # tuned ederek final test hatamıza ulaştıık.




















