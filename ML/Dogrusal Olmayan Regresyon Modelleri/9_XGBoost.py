# GBM'in hız ve tahmin performansını arttırmak üzere optimize edilmiş; ölçeklenebilir ve farklı platformlara entegre edilebilir halidir.
# R, Python, Hadoop, Scala, Julia ile kullanılabilir.
# Ölçeklenebilirdir.
# Hızlıdır.
# Tahmin başarısı yüksektir ve bir çok Kaggle yarışmasında başarısını kanıtlamıştır.
# Özetle GBM in hız ve performansı arttırılmış hali ve farklı platformlara entegre edilebilirliği olan bir modeldir.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost
from xgboost import XGBRegressor

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
xgb = XGBRegressor().fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred))) # ilkel test hatamız 355.465
############################## MODEL TUNNİNG ##############################
xgb_prarms = {'learning_rate': [0.1, 0.01, 0.5], 'max_depth': [2,3,4,5,8], 'n_estimators': [100, 200, 500, 1000],
              'colsample_bytree': [0.4, 0.7, 1]}
# learning_rate : öğrenme oranı, overfittingi engellemek için kullanılır.daraltma adım boyunu ifade etmektedir.
# Oluşturulacak ağaçda değişkenlenden alınacak alt küme oranını ifade ediyor.
xgb_cv_model = GridSearchCV(xgb, xgb_prarms, cv = 10, n_jobs=-1, verbose=2).fit(X_train, y_train)
print(xgb_cv_model.best_params_)
xgb_tuned = XGBRegressor(colsample_bytree = 0.4, learning_rate = 0.1,
                         max_depth = 2, n_estimators = 1000 ).fit(X_train, y_train)
y_pred = xgb_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # test hatamızı bulduk 367.85




















