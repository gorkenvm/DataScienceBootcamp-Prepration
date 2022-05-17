import pandas as pd
pd.options.display.width = 0
import seaborn as sns
import matplotlib.pyplot as plt
planets = sns.load_dataset("planets")
df = planets.copy()

df_num = df.select_dtypes(include= ["float64","int64"])
print(df_num.head())
print("---------------------------")
print(df_num.describe().T)              # Verinin tamamı için describe
print("---------------------------")
print(df_num["distance"].describe())    # Sadece distance için describe
print("---------------------------")

## İsterseniz Aşağıdaki gibi describe benzer istediğiniz birşey yapabilirsiniz.
print("Ortalama: " + str(df_num["distance"].mean()))
print("Dolu Gözlem Sayısı: " + str(df_num["distance"].count()))
print("Maksimum Değer: " + str(df_num["distance"].max()))
print("Minimum Değer: " + str(df_num["distance"].min()))
print("Medyan: " + str(df_num["distance"].median()))
print("Standart Sapma: " + str(df_num["distance"].std()))

# Bunu bir method ile yazarak kendi describe'ınızı oluşturabilirsiniz.










