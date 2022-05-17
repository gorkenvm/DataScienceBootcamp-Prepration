import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
import pandas as pd
pd.options.display.width = 0
import seaborn as sns

diamonds = sns.load_dataset('diamonds')
df = diamonds.copy()
print(df.head())
# price'ın dağılımını ve yoğunluğunun nerelerde olduğunu inceledik.
sns.displot(df.price, kde = False)
plt.show()

sns.displot(df.price, kde = True)
plt.show()
# Sadece yoğunluk olsun ve içide dolu olsun istedigimizde;
sns.kdeplot(df.price, shade = True)
plt.show()
# Grafigin bizden gizlediği şeyleri görebilmek için
# Çaprazlamalar yapacağız.
# Bağımlı değişken price, ana değişkenimiz.
(sns
 .FacetGrid(df,
               hue = "cut",
               height = 5,
               xlim = (0,10000))
 .map(sns.kdeplot, "price", shade = True)
 .add_legend()
);
plt.show()


sns.catplot(x = "cut", y = "price", hue = "color", kind = "point", data = df)
plt.show()












