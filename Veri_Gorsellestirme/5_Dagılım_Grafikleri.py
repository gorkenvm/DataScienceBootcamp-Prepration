import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
pd.options.display.width = 0

diamonds = sns.load_dataset("diamonds")
df = diamonds.copy()
print(df.head())
# Dağılım Grafikleri

# Bar Plot
# Sayısal verilere bakmadan Görsel ile yorum yapmamıza yardımcı olur.
# Bar plot: Kategorik verilerin görselleştirilmesi için kullanılır.

# Veri Setinin Hikayesi
"""
Mücevherler ve pırlantaların özelliklerini sunar

price: dolar cinsinden fiyat (326-18.823)
carat: ağırlık (0,2-5,01)
cut: kalite (Fair, Good, Very Good, Premium, Ideal)
color:  renk kategorik değişken, kalite bilgisi verir. (from j(worst) to D(best))
clarity: kalite, temizliği (I1 (worst), SI2,SI1,VS2,VS1,VVS2,VVS1,IF(best))
x: length in mm (0-10.74) yapısal özellik
y: width in mm (0-58.9)
z: depth in mm (0-31.8)
depth: toplam derinlik yüzdesi = z / mean(x,y) = 2*z/(x+y)(43-79)
table: elmasın en geniş noktasına göre genişliği (43-95)
"""

print(df.info())
print(df.describe().T)                  # İnfo ve describe ile numeric verileri inceleyip fikir sahibi olduk.

# Kategorik verileri inceleyeceğiz.
print(df["color"].value_counts())
print(df["cut"].value_counts())
print(df["clarity"].value_counts())
# Kategorik verileri görselleştireceğiz fakat bunlar nominal değil yani ordinal'ler
# Bu yüzden bunları tanımlamamız lazım.
from pandas.api.types import CategoricalDtype

print(df.cut.head())
df.cut = df.cut.astype(CategoricalDtype(ordered=True)) # Kategorik bir değişkendir hatta sıralıdır diye tanıttık.
print(df.dtypes)
print(df.cut.head(1))
# !!!!! ÖNEMLİ : ['Ideal' < 'Premium' < 'Very Good' < 'Good' < 'Fair']
# Çıktı yukarıdaki gibi ama biz verinin hikayesinden biliyoruz ki sıralama bu şekilde değildir.
cut_kategoriler = ["Fair", "Good", "Very Good","Premium", "Ideal"] # fonksiyona vermek üzere gerçek sıralamayı yazdık.
df.cut = df.cut.astype(CategoricalDtype(categories= cut_kategoriler, ordered=True)) # categories kısmına listeyi verdik.
print(df.cut.head(1))

### BAR PLOT ###
import seaborn as sns
import pandas as pd
(df["cut"]
 .value_counts()
 .plot.barh()
 .set_title("Cut Değişkeninin Sınıf Frekansı"));    # set_title özelliği ile başlık yazdırıldı.
# Noktalar başta olacak şekilde enter'layınca güzel görünüyor. Özellikle başlıkların kaymaması için yapılır.
plt.show()

sns.barplot(x = "cut", y= df.cut.index, data = df); # x hangi column, y index data veri bilgileri veriyoruz.
plt.show()                                          # görseli çok daha güzel ve kullanıma uygun.






