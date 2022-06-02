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

import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = 20
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

##################### MODEL VE TAHMİN ####################
gbm_model = GradientBoostingClassifier().fit(X_train,y_train)
y_pred = gbm_model.predict(X_test)
print(accuracy_score(y_test,y_pred))    # 0.744

##################### MODEL TUNİNG ####################
gbm = GradientBoostingClassifier()
gbm_params = {'learning_rate': [0.1,0.01,0.001,0.05], "n_estimators": [100,300,500,1000], "max_depth": [2,3,5,8]}
gbm_cv_model = GridSearchCV(gbm, gbm_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
print(gbm_cv_model.best_params_)

### FİNAL MODEL
gbm_tuned = GradientBoostingClassifier(learning_rate=0.01, max_depth=5, n_estimators=500).fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred ))

#### DEĞİŞKEN ÖENM DÜZEYLERİ
feature_imp = pd.Series(gbm_tuned.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y = feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title('Değişken Önem Düzeyleri')
plt.show()



