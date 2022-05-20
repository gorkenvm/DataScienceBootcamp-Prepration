# Basit Doğrusal Regresyon : 1 tane Bağımsız ve 1 tane Bağımlı değişkenden oluşan regresyon modellerdir.
# AMAÇ : Bağımlı ve Bağımsız değişken arasındaki ilişkiyi ifade eden doğrusal fonksiyonu bulmaktır.
# Modellemek : İlişkileri belirli matematiksel formlarda ifade etmek demektir.
# Yi = b0 + b1xi Günün sonunda ulaşacağımız şey böyle bir fonksiyon olacaktır.
# Emlak örneğini düşünürsek; Yi evin fiyatı, b0 bulunan bir sabit, b1 bulunan bir parametre, xi evin metrekaresi
import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Advertising.csv")
df = df.iloc[:,1:] # ilk columnda işimize yaramayacak sıra numaraları vardı sildik.
print(df.head())
# Satış verisi var ve Tv Radio Newspaper reklam harcamaları var
# Amacımız 1 tane bağımsız değişken alarak sales'e etkisini inceleyeceğiz
print(df.info())
import seaborn as sns
import matplotlib.pyplot as plt
# Tv ve sales arasında ki ilişkiyi gözlemlemek için görselleştirme yapacagız.
#sns.jointplot(x = "TV", y = "sales", data = df, kind = "reg")
#plt.show()
# Şimdi bu doğrusal regresyonu modelleyeceğiz.

                    ####### MODEL KURMA #######
from sklearn.linear_model import LinearRegression
X =df[["TV"]]       # X'e bağımsız değişkenimiz olan Tv'yi atadık
print(X.head())
y = df[["sales"]]   # y'ye bağımlı değişkenimiz olan sales'i atadık
print(y.head())
reg = LinearRegression()    # Model Nesnesi Oluşturduk.
model = reg.fit(X, y)       # Modeli fit et yani kur dedik

print(dir(model))           # Bu nesnenin içinden alabileceğimiz bazı bilgiler bize sunulmuş oldu.
print(model.intercept_)     # b0 katsayısı
print(model.coef_)          # b1 katsayısı
# b0 ve b1 gibi katsayıları bulmak için çeşitli yöntemler var. sklearn ile parametre bulma yöntemlerini kullanacağız.
# rkare yi verir score, bu çok önemli birşey
print(model.score(X, y))    # nedir rkare ? Bağımlı değişkendeki değişikliğin, bağımsız değişkenlerce açıklanması yüzdesidir.
# Yani sales'taki değişimin %61'i TV reklamlarından kaynaklanıyor bilgisini aldık.

                        ####### TAHMİN #######
# Kurduğumuz modelin görsel olarak nasıl göründüğünü gösterelim
g = sns.regplot(df["TV"], df["sales"], ci = None, scatter_kws = {'color':'r','s':9})    # scatter_kws = noktaların rengi ve büyüklüğü, ci= doğru etrafına güven aralığı koyma
g.set_title("Model Denklemi : Sales = 7.03 + TV*0.047")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10,310)                   # Daha güzel görünmesi için yapıldı
plt.ylim(bottom = 0)                # Daha güzel görünmesi için yapıldı
#plt.show()
# Model çıktı nasıl tahmin ettireceğiz?
# Bu yıl TV harcamasına ne kadar yatırım yaparsam satışım ne kadar olur ? Bu soruyu cevaplayacağız.
print(model.predict([[165]])) # predict tahmin et demek, en çok kullanacağımız fonksiyonlardan biri olacaktır.
# 165 birim TV harcaması yaparsak ne kadar sales olur? 14.876 cevabımız.
yeni_veri = [[5],[15],[30]]     # 3 farklı departmandan rakamlar verip sordular
print(model.predict(yeni_veri))

                    ####### ARTIKLAR (HATALAR) #######
# Bu konu modelleme ve tahmini anladıktan sonra en önemli konudur.
# İşin kalitesini belirleyen kısım budur. Yani hataları en düşük seviyeye getirmektir.
# Şimdi manuel olarak hataları bulalım.
gercek_y = y[:10]   # y, bağımsız değerimiz. dataframe halinde
tahmin_edilen_y = pd.DataFrame(model.predict(X)[:10])   # Tahmin edilen değerlerdende ilk 10'u aldık ve dataframe'e çevirdik.
hatalar = pd.concat([gercek_y, tahmin_edilen_y], axis=1)  # her iki dataframe'i birleştirdik
hatalar.columns = ["gercek_y", "tahmin_edilen_y"]       # column islemlendirmesini yaptık
print(hatalar)  # Gözlemleyelim
hatalar["hata"] = hatalar["gercek_y"] - hatalar["tahmin_edilen_y"]  # Çıkarma işlemi ile hatayı bulduk
print(hatalar)  # negatifler gitsin diye kare alalım
hatalar["hata_kareler"] = hatalar["hata"] ** 2
print(hatalar)  # Gözlemleyelim
# hata kareler ortalamasını alalım
np.mean(hatalar["hata_kareler"])    # 10 gözlem üzerinden manuel olarak MSE hesapladık.
# Bu rakama göre modelimiz şu kadar başarılı diyebiliriz.
