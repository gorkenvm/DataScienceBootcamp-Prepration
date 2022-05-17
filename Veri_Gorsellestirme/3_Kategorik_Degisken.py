import pandas as pd
pd.options.display.width = 0
import seaborn as sns
import matplotlib.pyplot as plt
planets = sns.load_dataset("planets")
df = planets.copy()

# Sadece Kategorik Değişkenler ve Özetleri
kat_df = df.select_dtypes(include= ["object"])  # df de data tipine göre seç, kategorik'i seç
print(kat_df.head())
print("----------------------------------------")
print(kat_df.method.unique())   # Method'ta ki kategorik değişkenlere bakıyoruz.
# Baya çokmuş gözle anlayamayız toplatalım hepsini
print("----------------------------------------")

## Kategorik Değişkenin Sınıflarına ve Sınıf Sayısına Erişmek
print(kat_df["method"].value_counts().count())  # Toplam 10 tane kategori var
print("---------------------------------------")
print(kat_df["method"].value_counts())  # her bir kategoride kaç tane değer vardır.
print("---------------------------------------")
df["method"].value_counts().plot.barh()    # barh yerine bar yazarsanız dikey barlar plot eder.
plt.show()














