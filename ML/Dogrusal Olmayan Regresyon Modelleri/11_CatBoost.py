# Kategorik değişkenler ile otomatik olarak mücadele edebilen, hızlı, başarılı bir diğer GBM türevidir.
# Sayısal değişkenleride kategorik olarak değiştirebildiğimizden çok fazla kategorik değişken olabiliyor
## Ağaç oluşturulurken bu kategorilerin kırılımları kolaylaştırmasını sağlıyor.
# Hızlı ve ölçeklenebilir GPU desteği
# Daha başarılı tahminler
# Hızlı train ve hızlı tahmin

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from catboost import CatBoostRegressor

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
catb_model = CatBoostRegressor().fit(X_train, y_train)
y_pred = catb_model.predict((X_test))
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # 351.194
############################## MODEL TUNNİNG ##############################
catb_params = {'iterations': [200,500,100], 'learning_rate': [0.01,0.1],
               'depth':[3,6,8]}
# iteration = ağaç sayısı
catb_model = CatBoostRegressor()
catb_cv_model = GridSearchCV(catb_model, catb_params, cv = 5, n_jobs=-1,verbose=2).fit(X_train, y_train)
print(catb_cv_model.best_params_)
catb_tuned = CatBoostRegressor(depth=3, iterations=200,learning_rate = 0.1).fit(X_train, y_train)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # 351.19






