# Kategorik değişkenler ile otomatik olarak mücadele edebilen, hızlı, başarılı bir diğer GBM türevidir.
# Sayısal değişkenleride kategorik olarak değiştirebildiğimizden çok fazla kategorik değişken olabiliyor
## Ağaç oluşturulurken bu kategorilerin kırılımları kolaylaştırmasını sağlıyor.
# Hızlı ve ölçeklenebilir GPU desteği
# Daha başarılı tahminler
# Hızlı train ve hızlı tahmin

import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = 20
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from catboost import CatBoostClassifier

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

##################### MODEL VE TAHMİN ####################
catb_model = CatBoostClassifier().fit(X_train,y_train)
y_pred = catb_model.predict(X_test)
print(accuracy_score(y_test,y_pred))

##################### MODEL TUNİNG ####################
catb = CatBoostClassifier()
catb_params = {"iteration": [200,500,1000], "learning_rate": [0.1,0.03],"depth":[5,8]}
catb_cv_model = GridSearchCV(catb,catb_params,cv=5,n_jobs=-1,verbose=2).fit(X_train,y_train)
print(catb_cv_model.best_params_)
### FİNAL MODEL
catb_tuned = CatBoostClassifier(depth=8, iterations=2, learning_rate=0.03).fit(X_train,y_train)
y_pred = catb_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))

#### DEĞİŞKEN ÖNEM DÜZEYLERİ
feature_imp = pd.Series(catb_tuned.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y = feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title('Değişken Önem Düzeyleri')
plt.show()



