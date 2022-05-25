# Amaç: veri seti içerisindeki karmaşık yapıları basit karar yapılarına dönüştürmektedir.
# Heterojen veri setleri belirlenmiş bir hedef değişkene göre homojen alt gruplara ayrılır.
# Sınıflandırma veya regresyon problemlerinde kullanılır
# Gözetimlidir. Şimdiye kadar öğrendiklerimiz hepsi böyleydi.
# Aşırı öğrenmeye meyillidir. Büyük veriler için çok uygun değildir.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from warnings import filterwarnings # Uyarıların çıkmasını engeller.
filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Hitters.csv")
df = df.dropna() # Eksik değerleri uçurduk, konu kapsamında olmadığından dolayı
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])   # Kategorik değişkenleri one-hot-encoding yaklaşımı ile dummy'e çevirdik
y = df["Salary"]        # Bağımlı değişkenimiz.
X_ = df.drop(['Salary','League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)

############################## TEK DEĞİŞKENLİ MODEL ve TAHMİN ##############################
# Değişken seçiyoruz. Atış sayısını seçtik
X_train = pd.DataFrame(X_train["Hits"])
X_test = pd.DataFrame(X_test["Hits"])
cart_model = DecisionTreeRegressor(max_leaf_nodes=10) # max yaprak sayısını belirliyoruz.
cart_model.fit(X_train, y_train)
# Agaç yapısını gözlemlemek adına bir grafik oluşturalım.
#-------------------------------------------------------------------
X_grid = np.arange(min(np.array(X_train)),max(np.array(X_train)), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, cart_model.predict(X_grid), color = 'blue')
plt.title('CART REGRESYON AGACI')
plt.xlabel('Atış Sayısı(Hits)')
plt.ylabel('Maaş(Salary)')
plt.show()
#-------------------------------------------------------------------
y_pred = cart_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))

############################## TÜM DEĞİŞKENLİ MODEL ve TAHMİN ##############################
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Hitters.csv")
df = df.dropna() # Eksik değerleri uçurduk, konu kapsamında olmadığından dolayı
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])   # Kategorik değişkenleri one-hot-encoding yaklaşımı ile dummy'e çevirdik
y = df["Salary"]        # Bağımlı değişkenimiz.
X_ = df.drop(['Salary','League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)

cart_model = DecisionTreeRegressor(max_leaf_nodes=10) # max yaprak sayısını belirliyoruz.
cart_model.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))  # 486 dan 450 ye düştü

# Çıkarılabilecek sonuçlar
# Seçtiğimiz değişkenin oldukça yüksek açıklanabilirlik sağladığı
# Yeni değişkenler eklendikçe daha başarılı tahminler yapılabilmektedir.













