# XGBoost'un eğitim süresi performansını arttırmaya yönelik geliştirilen bir dğierGBM türüdür.
# Veri ve değişken sayısı arttıkça XGBoost yavaş kalabilmektedir. bu yüzden Light GMB kullanacağız.

import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = 20
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from lightgbm import LGBMClassifier

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

##################### MODEL VE TAHMİN ####################
lgbm_model = LGBMClassifier().fit(X_train,y_train)
y_pred = lgbm_model.predict(X_test)
print(accuracy_score(y_test,y_pred))    # 0.722

##################### MODEL TUNİNG ####################
lgbm = LGBMClassifier()
lgbm_params = {"learning_rate":[0.1,0.001,0.01], "n_estimators": [100,500,200], "max_depth":[1,2,35,8]}
lgbm_cv_model = GridSearchCV(lgbm,lgbm_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
#### FİNAL MODEL
lgbm_tuned = LGBMClassifier(learning_rate=0.01, max_depth=1, n_estimators=500).fit(X_train,y_train)
y_pred = lgbm_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))    # 0.757

#### DEĞİŞKEN ÖNEM DÜZEYLERİ
feature_imp = pd.Series(lgbm_tuned.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y = feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title('Değişken Önem Düzeyleri')
plt.show()




