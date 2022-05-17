import pandas as pd
pd.options.display.width = 0
import seaborn as sns

planets = sns.load_dataset("planets")
df = planets.copy()
# Eksik, aykırı degerleri veri ön işlemede detaylı ele alacağız.
# Şimdilik hızlıca dokunup bazı basit çözümler uygulayacagız.

# Eksik gözlem var mı ?
print(df.isnull().values.any()) # Veride hiç eksik değer var mı *
print("--------------------------------------------------")
# Hangi değişkende kaçar tane eksik var
print(df.isnull().sum())    # eksik sayıların toplamı
"""
method ve number da eksik veri yok, zaten number da eksik veri olsa yani gezegen yoktur olsa ozmn galakside olmaz.
orbital_period = 43 eksik var, bu eksikliğin yapısı nasıl ortaya cıktı bunu genıs genıs ele alacagız.
"""













