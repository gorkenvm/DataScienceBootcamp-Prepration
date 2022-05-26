# Topluluk Öğrenme Yöntemleri : Birden fazla algoritmanın ya da birden fazla ağacın bir araya gelerek toplu bir şekilde tahmin etmeye çalışmasıdır.
# Bootstrap rastgele örnekleme yöntemi : gözlem birimlerinin içinden yerine koymalı bir şekilde tekrar tekrar örnek çekmek demektir.
# Bagging Çalışma Prensibi : Elimizde 1000 gözlemli bir veri var, 750 tane rastgele gözlem çekiliyor (bootstrap)
# bununla bir ağaç oluşturuluyor. 750 gözlem yerine konularak tekrar rastgele 750 çekiliyor ve ağaç oluşturuluyor.
# T adet ağaç oluşturuluyor ve daha fazla tahmin değeri elde ediyoruz.
# RMSE yi düşürüyor, doğru sınıflandırma oranını arttırıyor, varyansı düşürür ve ezberlemeye karşı dayanıklıdır.

# Random Subspace : Değişkenlerin rastgele seçilmesi
## Random Forest : Temeli birden çok karar ağacın ürettiği tahminlerin bir araya getirilerek değerlendirilmesine dayanır.
# Bagging yöntemi + Random Subspace'dir, hem gözlemleri(rowları) hem de değişkenleri(columnları) rastgele seçerek başarı oranını arttırır.
# Ağaç oluşturmada veri setinin 2/3 ü kullanılır. Dışarıda kalan veri ile ağaçların performansı değerlendirilir ve değişken önemi belirlenir.
# Her düğüm noktasında rastgele değişken seçimi yapılır. ( regresyonda değişken sayısı/3, sınıflandırmada ise değişken sayısının karekökü)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from warnings import filterwarnings # Uyarıların çıkmasını engeller.
filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Hitters.csv")
df = df.dropna() # Eksik değerleri uçurduk, konu kapsamında olmadığından dolayı
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])   # Kategorik değişkenleri one-hot-encoding yaklaşımı ile dummy'e çevirdik
y = df["Salary"]        # Bağımlı değişkenimiz.
X_ = df.drop(['Salary','League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)

############################## MODEL ve TAHMİN ##############################
rf_model = RandomForestRegressor(random_state= 42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred))) # ilkel hatamız 344.819
############################## MODEL TUNİNG ##############################
# Random Forest için Optimize edilebilecek bir çok parametre mevcuttur. En önemlileri
# 1- n_estimators, 2- max_features(değişkensayısı), 3- min_sample_leaf ve min_sample_split
rf_params = {"max_depth": [5,8,],
             "max_features": [2,10],
             "n_estimators": [200,500,2000],
             "min_samples_split": [2,10,100]}   # Çok uzun sürdüğü için azalttım listelerin içeriklerini, makinanız güçlüyse daha fazla seçenek ekleyebilirsiniz
rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs= -1, verbose= 2).fit(X_train, y_train)
print(rf_cv_model.best_params_)
# En iyi parametreler {'max_depth': 8, 'max_features': 2, 'min_samples_split': 2, 'n_estimators': 200} bunları kullanarak tuned edelim
rf_model = RandomForestRegressor(random_state=42, max_depth = 8, max_features = 2, min_samples_split = 2, n_estimators = 200 )
rf_tuned = rf_model.fit(X_train, y_train) # Tuned model kuruldu
y_pred = rf_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # ilginç bir şekilde test hatamız arttı. 349.164

### Değişken Önem Düzeyi
Importance = pd.DataFrame({'Importance': rf_tuned.feature_importances_*100}, index= X_train.columns)
Importance.sort_values(by= 'Importance', axis= 0, ascending= True).plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.gca().legend = None
plt.show()






