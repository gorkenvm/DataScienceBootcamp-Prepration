# Regresyondakinin aynısıdır.


import numpy as np
import pandas as pd
pd.options.display.max_columns = 20
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)

##################### MODEL VE TAHMİN ####################
mlpc_model = MLPClassifier().fit(X_train, y_train)
print(mlpc_model.coefs_)    # Katsayılar
# Sınıflandırma için activation='logistic', doğrusal için 'relu'
# solver= 'adam' büyük veri seti için, 'lbfgs' küçük veri seti için
y_pred = mlpc_model.predict(X_test)
print(accuracy_score(y_test, y_pred)) # ilkel test hatası 0.705

##################### MODEL TUNİNG ####################
mlpc_params = {'alpha':[1,5,0.1,0.01,0.03,0.005,0.0001],
               'hidden_layer_sizes': [(10,10),(100,100,100),(100,100),(3,5)]}
mlpc = MLPClassifier(solver='lbfgs', activation='logistic')
mlpc_cv_model = GridSearchCV(mlpc, mlpc_params, cv=10, n_jobs=-1,verbose=2).fit(X_train,y_train)
print(mlpc_cv_model.best_params_)

##### FİNAL MODEL
mlpc_tuned = MLPClassifier(solver='lbfgs', activation='logistic',alpha=1,hidden_layer_sizes=(3,5)).fit(X_train,y_train)
y_pred = mlpc_tuned.predict(X_test)
print(accuracy_score(y_test, y_pred))





