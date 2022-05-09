import seaborn as sns

df = sns.load_dataset("planets")
print(df.head())    # İLK 5 ROW
print("--------------------------")
print(df.shape)     # BOYUT BİLGİSİ
print("--------------------------")
print(df.mean())    # ORTALAMA
print("--------------------------")
print(df["mass"].mean())    # İSTEDİĞİMİZ COLUMN SEÇİPTE YAPABİLİRİZ.
print("--------------------------")
print(df["mass"].count())   # SAYMA, KAÇ ADET VAR.
print("--------------------------")
print(df["mass"].min())     # MİNİMUM DEĞER
print("--------------------------")
print(df["mass"].max())     # MAXİMUM DEĞER
print("--------------------------")
print(df["mass"].sum())     # TOPLAMI VERİR.
print("--------------------------")
print(df["mass"].std())     # STANDART SAPMAYI VERİR.
print("--------------------------")
print(df["mass"].var())     # VARİANCE I VERİR.
print("--------------------------")

## HERBİRİNİ AYRI AYRI YAPMAK ZORUNDA DEĞİLİZ.
# DESCRİBE FONKSİYONU HEPSİNİ BİR ARADA GETİRİYOR.
print(df.describe())
print("--------------------------")
print(df.describe().T)                  # DAHA İYİ GÖRÜMMESİ İÇİN T(TRANSPOSE)SİNİ ALIYORUZ.
print("--------------------------")
print(df.dropna().describe().T)         # EKSİK VERİLERİ GÖZ ÖNÜNE ALMADAN DESCRİBE ET DEMEK.
