# XGBoost'un eğitim süresi performansını arttırmaya yönelik geliştirilen bir dğierGBM türüdür.
# Veri ve değişken sayısı arttıkça XGBoost yavaş kalabilmektedir. bu yüzden Light GMB kullanacağız.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

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
lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # 363.871
############################## MODEL TUNNİNG ##############################
lgbm_params = {'learning_rate': [0.01, 0.1, 0.5, 1], 'n_estimators': [20,40,100,200,500,1000],
                'max_depth':[1,2,3,4,5,6,7,8,9,10]}
lgbm_cv_model = GridSearchCV(lgb_model,lgbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
print(lgbm_cv_model.best_params_)
lgbm_tuned = LGBMRegressor(learning_rate=0.1, max_depth=6, n_estimators= 20).fit(X_train,y_train)
y_pred = lgbm_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # 371.504



















