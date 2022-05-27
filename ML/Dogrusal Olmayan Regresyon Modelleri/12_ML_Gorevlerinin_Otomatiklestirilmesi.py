# Tüm algoritmalarda aynı veri setini kullandık
# Aynı bölümleri tekrar tekrar değilde otomatik olarak yaptıracağız.

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
from lightgbm import LGBMRegressor
import xgboost
from xgboost import XGBRegressor

from warnings import filterwarnings # Uyarıların çıkmasını engeller.
filterwarnings('ignore')

##### VERİ YÜKLEME VE ÖNİŞLEME İŞLEMLERİ #####
import numpy as np

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Hitters.csv")
df = df.dropna() # Eksik değerleri uçurduk, konu kapsamında olmadığından dolayı
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])   # Kategorik değişkenleri one-hot-encoding yaklaşımı ile dummy'e çevirdik

models = [LGBMRegressor, XGBRegressor, GradientBoostingRegressor, RandomForestRegressor, DecisionTreeRegressor,
                  MLPRegressor, KNeighborsRegressor, SVR]
resRMSE = {}
##### OTOMOTİZE ETME ######
def compML(df, y='Salary', alg=models):    # df dataset, y bağımlı değişken, alg algoritma
    # Train Test ayrımı
    y = df[y]  # Bağımlı değişkenimiz.
    X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
    X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    for alg in models:
        # Modelleme
        model = alg().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
        model_ismi = alg.__name__
        resRMSE[model_ismi] = RMSE
    print(resRMSE)

compML(df)

# Yukarıda özetle; compML sınıfına df = dataset, y = bağımlı değişken, alg= algoritmalar tanımladık.
# Varsayılan değer olarak y 'Salary' alg ise models listesidir dedik.
# bağımlı bağımsız değişken, train split kısmını tamamladık.
# for döngüsü ile her bir liste değeri için model oluşturduk.
# model ismini print edebilmek için __name__ yapısını kullandık.
# model isim ve RMSE leri boş bir sözlüğe ekledik.
