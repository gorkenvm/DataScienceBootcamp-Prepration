# Diğer bölümlerden farklı olarak standartlaştırma işlemi gerçekleştireceğiz.
# ML algoritmaların hemen hemen hepsi standartlaştırmayı normalde sever
# Fakat bazı algoritmalar heterojen veri setlerinde iyi çalışırken bazıları homojen veri setlerinde iyi çalışır.
# Yapay sinir ağları homojen veri setlerinde daha iyi çalışır.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

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
### Standartlaştırm
scaler = StandardScaler()   # Standartlaştırıcı oluşturuldu.
scaler.fit(X_train)         # train seti standartlaştırıldı.
X_train_scaled = scaler.transform(X_train) # X train ölçeklendirildi.
X_test_scaled = scaler.transform(X_test) # X test ölçeklendirildi.
### Model kurma
mlp_model = MLPRegressor().fit(X_train_scaled, y_train)
print(mlp_model.predict(X_test_scaled)[:5] )   # bağımsız test için tahmin et
y_pred = mlp_model.predict(X_test_scaled)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # ilkel hata değerimiz. 662
### parametreleri değiştirelim
mlp_params = {"alpha": [0.1, 0.01, 0.02, 0.001, 0.0001], "hidden_layer_sizes": [(10,20),(5,5), (100,100)]} # (10,20) 2 katman olsun 10 ve 20 nöron olsun
mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv= 10, verbose= 2, n_jobs= -1).fit(X_train_scaled, y_train)
print(mlp_cv_model.best_params_)
#### Final Model
mlp_tuned = MLPRegressor(alpha= 0.02, hidden_layer_sizes=(100,100)).fit(X_train_scaled, y_train)
y_pred = mlp_tuned.predict(X_test_scaled)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # 662 den 358 e düşürdük.



























