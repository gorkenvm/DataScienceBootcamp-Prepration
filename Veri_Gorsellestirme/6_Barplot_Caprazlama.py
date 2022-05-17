import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
import pandas as pd
pd.options.display.width = 0

diamonds = sns.load_dataset("diamonds")
df = diamonds.copy()
# Kategoriler
cut_kategoriler = ["Fair","Good","Very Good","Premium","Ideal"]
color_kategoriler = ["J", "I", "H", "G", "F", "E", "D"]
clarity_kategoriler = ["I1","SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
# Kategorige Dönüştürme
df.cut = df.cut.astype(CategoricalDtype(categories = cut_kategoriler, ordered = True))
df.color = df.color.astype(CategoricalDtype(categories = color_kategoriler, ordered = True))
df.clarity = df.clarity.astype(CategoricalDtype(categories = clarity_kategoriler, ordered = True))

print(df.head())
print(df.info())
print(df.describe().T)

#### ÇAPRAZLAMA ####
# Kırılımları görme, değişkenlerin etkilerinin birlikte gösterilmesi denilebilir.
# Somut analitik gözlemler yapabilmemize yardımcı olacaktır.
# Bilgi çıkaracağımız bir bölümdür.

sns.catplot(x = "cut", y = "price", data = df);
plt.show()
# yoğunluklara baktıgımızda; Fair 7.500 altında yogun üstünde çok seyrek, ve kalite arttıkça yukarı fiyatlardaki
# yoğunlukta artmaktadır. Bu beklediğimiz birşeydi. Grafigin bizden sakladıklarını görebilmek için
# Çaprazlama yapacağız.
sns.barplot(x = "cut", y = "price", hue = "color", data = df)
plt.show()
sns.barplot(x = "cut", y = "carat", hue = "color", data = df)
plt.show()

print(df.groupby(["cut","color"])["price"].mean())  # barplotlarda y ekseninde bozulma var gibi görünüyor.
# Bozulma olmadığını doğruluyoruz buradan. iki kategoriye göre gruplayınca priceları düşük.
