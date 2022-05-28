# Amaç sınıflandırma problemi için bağımlı ve bağımsız değişken arasındaki ilişkiyi tanımlayan doğrusal bir model kurmaktır.
# Çoklu doğrusal regresyonun sınıflandırma problemlerine uyarlanmış fakat ufak farklılıklara tabi tutulmuş bir versiyon olarak düşünebiliriz.
# Adını bağımlı değişkene uygulanan 'Logit' dönüşümünden alır.
# Bağımsız değişken değerleri kullanıldığında Bağımlı değişkenin 1 olarak tanımlanan değerinin gerçekleşme olasılığı hesaplanır. Dolayısıyla bağımlı değişkenin alacağı değerler ile ilgilenilmez.

import pandas as pd
pd.options.display.max_columns = 20
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# VERİ SETİ HİKAYESİ VE PROBLEM: ŞEKER HASTALIĞI TAHMİNİ
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\diabetes.csv")
print(df.head())
print(df["Outcome"].value_counts())
print(df.describe().T)
# Bağımlı bağımsız değişken atama
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
# Train test diye ayırmadık ama ayırmamız gerekiyor. Siz yapabilirsiniz.

##################### MODEL KURMA ####################

loj_model = LogisticRegression(solver="liblinear").fit(X, y)    # Model kuruldu
print(loj_model.intercept_) # Sabit
print(loj_model.coef_)      # Katsayılar
print(loj_model.predict(X)[:10])    # ilk 10 gözlem için tahmin
y_pred = loj_model.predict(X)       # Tahmin edilen y ler
print(confusion_matrix(y, y_pred))  # karmaşıklık matrisi ile hata değerlendirmesi yapacağız.
print(accuracy_score(y, y_pred))    # Başarılı yaptığımız bölü tüm durumlar, doğruluk oranıdır.
print(classification_report(y, y_pred)) # accuracy_score'a göre daha detaylı sonuç verir.
#loj_model.predict_proba(X) # eğer olasılıklarını istiyorsak bunu kullanabiliriz. Çok ileri ihtiyaçlar olmadıkça kullanmıyoruz.

# roc eğrisi için kod parçası
logit_roc_auc = roc_auc_score(y, loj_model.predict(X))
fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label = 'AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.savefig('Log_ROC')
plt.show()

##################### MODEL TUNNİNG ####################
# Model tunning değil model validation yapacağız.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
loj_model = LogisticRegression(solver="liblinear").fit(X_train, y_train)    # Model kuruldu
y_pred = loj_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(cross_val_score(loj_model, X_test, y_test, cv=10).mean())
# cross_val_score ile valide ediyoruz. cv yapmak zorunda değilsin ama yaparsan daha iyi
# Normalde train için bunları yapacaz ve test seti ile de test edeceğiz



