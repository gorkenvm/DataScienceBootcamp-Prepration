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
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = 20
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

##################### MODEL VE TAHMİN ####################
rf_model = RandomForestClassifier().fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
print(accuracy_score(y_test,y_pred))    # 0.744

##################### MODEL TUNİNG ####################
rf = RandomForestClassifier()
rf_params = {'n_estimators': [100,200,500,1000], "max_features":[3,5,7,8],
             "min_samples_split": [2,5,10,20]}
rf_cv_model = GridSearchCV(rf,rf_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
print(rf_cv_model.best_params_)

#### FİNAL MODEL
rf_tuned = RandomForestClassifier(max_features= 8, min_samples_split= 10, n_estimators= 200).fit(X_train,y_train)
y_pred = rf_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))    # 0.740

#### DEĞİŞKEN ÖENM DÜZEYLERİ
feature_imp = pd.Series(rf_tuned.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y = feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title('Değişken Önem Düzeyleri')
plt.show()




