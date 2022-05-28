# Amaç: İki sınıf arasındaki ayrımın optimum olmasını sağlayacak hiper düzlemi bulmaktır.
# Sınıf aralığının maksimum uzaklıkta olması sağlanır. Sınıflar arasına doğru koyuyoruz.

# Doğrusal olmayan SVM de ise bir boyut daha eklenerek 3 boyutlu görüntü elde ediliyor ve araya düzlem konularak sınıflar ayrılıyor.
# Kernel trick deniliyor bu işleme

import numpy as np
import pandas as pd
pd.options.display.max_columns = 20
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.svm import SVC


df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

##################### MODEL VE TAHMİN ####################
svm_model = SVC(kernel='linear').fit(X_train, y_train) # kernel= linear, kernel=rbf doğrusal ve doğrusal olmayan olarak seçebiliyoruz
y_pred = svm_model.predict(X_test)
print(accuracy_score(y_test, y_pred))

##################### MODEL TUNİNG ####################
svm = SVC()
svm_params = {'C': np.arange(1,10), 'kernel': ['linear', 'rbf']}
svm_cv_model = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
print(svm_cv_model.best_score_)
print(svm_cv_model.best_params_)

#### FİNAL MODEL
svm_tuned = SVC(C=2, kernel='linear').fit(X_train, y_train)
y_pred = svm_tuned.predict(X_test)
print(accuracy_score(y_test, y_pred))





