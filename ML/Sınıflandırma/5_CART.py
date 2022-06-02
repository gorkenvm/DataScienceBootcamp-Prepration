# Amaç: Veri seti içerisindeki karmaşık yapıları basit karar ağaç yapılarına dönüştürmektir.
# Heterojen veri setleri belirlenmiş bir hedef değişkene göre homojen alt gruplara ayrılır.
# Burada önemli olan ayırma işlemleridir.Dallanmalar

import numpy as np
import pandas as pd
pd.options.display.max_columns = 20
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

##################### MODEL VE TAHMİN ####################
cart_model = DecisionTreeClassifier().fit(X_train,y_train)
y_pred = cart_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
##################### MODEL TUNİNG ####################
cart = DecisionTreeClassifier()
cart_params = {'max_depth': [3,5,8,10], 'min_samples_split':[3,5,10,20,50]}
cart_cv_model = GridSearchCV(cart, cart_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
print(cart_cv_model.best_params_)

#### FİNAL MODEL
cart_tuned = DecisionTreeClassifier(max_depth=5, min_samples_split=20).fit(X_train,y_train)
y_pred = cart_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))




