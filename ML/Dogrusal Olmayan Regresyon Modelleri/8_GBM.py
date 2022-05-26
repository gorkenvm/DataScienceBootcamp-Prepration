# Adaboost(boosting) : Zayıf öğrenicileri bir araya getirerek güçlü bir öğrenici ortaya çıkarmak fikrine dayanır.
# Zayıf öğrenici nedir? RMSE si yüksek olanlardır.
# örneğin Random Forest'ta bir sürü ağaç oluşturuluyordu ve bazıları kötü sonuçlar veriyordu.
# Bu kötü sonuç verenleri bir araya getirerek bunlardan güçlü model çıkarmaktır adaboosting
# GBM : Adaboost'un sınıflandırma ve regresyon problemlerine kolayca uyarlanabilen geliştirilmiş versiyonudur.
# Artıklar üzerine tek bir tahminsel model formunda olan modeller serisi kurulur.
# Serideki bir model, bir öncekinin üzerine kurularak oluşturulur.
# GBM diferansiyellenebilen herhangi bir kayıp fonksiyonunu optimize edebilen Gradient descent algoritmasını kullanmaktadır.
# Bir çok temel öğreniciyi destekler (trees, linear terms, splines ...)
# Cost ve link fonksiyonları modifiye edilebilirdir.
# GBM aslında Boosting + Gradient Descent ten oluşur.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

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
gbm_model = GradientBoostingRegressor().fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred))) # ilkel hatamız 354.6346601810501

############################## MODEL TUNİNG ##############################
# criterion = 'friedman_mse'   bölünmelerle ilgili saflığı ifade etmektedir.
# learning_rate = 0.1   ağaçların katkısı ile ilgi ifadedir.
# loss = 'ls'   en küçük kareleri ifade etmektedir.
# subsample = 0.1  ağaç oluşturulurken göz önünde bulundurulan oranı ifade eder. 1 yazılırsa hepsini dahil ederek ağaç oluşturur.
gbm_params = {'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3,5,8], "n_estimators": [100, 200, 500],
              "subsample": [1, 0.5,0.8], "loss": ["ls","lad"]}
gbm_cv_model = GridSearchCV(gbm_model,gbm_params, cv=10, n_jobs=-1,verbose=2).fit(X_train, y_train)
gbm_tuned = GradientBoostingRegressor(learning_rate=0.1, loss='lad', max_depth=3, n_estimators=200, subsample=1).fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # test hatası 325.404 şuana kadar gördüğümüz en düşük test hatası

### Değişken Önem Düzeyi
Importance = pd.DataFrame({'Importance': gbm_tuned.feature_importances_*100}, index= X_train.columns)
Importance.sort_values(by= 'Importance', axis= 0, ascending= True).plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.gca().legend = None
plt.show()