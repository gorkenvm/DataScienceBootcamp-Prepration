import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = 20
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from xgboost import XGBClassifier

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

##################### MODEL VE TAHMİN ####################
xgb_model = XGBClassifier().fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
##################### MODEL TUNİNG ####################
xgb = XGBClassifier()
xgb_params = {"n_estimators": [100,500,1000], "subsample": [0.6,0.8,1],
              "max_depth":[3,5,7], "learning_rate":[0.1,0.001,0.01]}
xgb_cv_model = GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
print(xgb_cv_model.best_params_)

####FİNAL MODEL
xgb_tuned = XGBClassifier(learning_rate = 0.001, max_depth = 7, n_estimators = 500, subsample = 0.6).fit(X_train,y_train)
print(xgb_tuned.predict(X_test))
print(accuracy_score(y_test,y_pred))

#### DEĞİŞKEN ÖENM DÜZEYLERİ
feature_imp = pd.Series(xgb_tuned.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y = feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title('Değişken Önem Düzeyleri')
plt.show()













