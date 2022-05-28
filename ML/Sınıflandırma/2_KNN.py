# En yakın n sayıda ki komşuları inceleyerek sınıf bulunacaktır.
# Gözlemler arası uzaklık hesabı ile bulunur.

import numpy as np
import pandas as pd
pd.options.display.max_columns = 20
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

##################### MODEL VE TAHMİN ####################
knn_model = KNeighborsClassifier().fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print(accuracy_score(y_test, y_pred))   # 0.688

##################### MODEL TUNNİNG ####################
knn = KNeighborsClassifier()
knn_params = {"n_neighbors": np.arange(1,50)}
knn_cv_model = GridSearchCV(knn, knn_params, cv=10).fit(X_train, y_train)
print(knn_cv_model.best_score_)     # 0.748
print(knn_cv_model.best_params_)    # 11

### FİNAL MODEL
knn_tuned = KNeighborsClassifier(n_neighbors=11).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
print(accuracy_score(y_test, y_pred))   # 0.731 yükseldi, yükseldikçe başarı artıyor.












