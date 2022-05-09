import numpy as np
import pandas as pd

df = pd.DataFrame({'gruplar': ['A','B','C','A','B','C'],
                    'degisken1' : [12,23,33,22,11,99],
                    'degisken2': [100,253,333,262,111,969]},
                    columns = ['gruplar','degisken1','degisken2'])
print(df)
print("------------------------------")
# AGGREGATE
print(df.groupby('gruplar').aggregate(["min",np.median, max]))  # aggreagete ile birden fazla özellik yazabiliyorsun.
# min ve max pandasın içinde bu yüzden "" or '' içinde yazabilirsiniz veya yazmazsınız.
# fakat np.median pandas içinde olmadığı için "" or '' içinde yazamazsın
print("------------------------------")
print(df.groupby("gruplar").aggregate({"degisken1":"min","degisken2": "max"})) #degisk1 ve 2 için farklı şeyler uygulayacaksak.
# sözlük yapısından faydalanabiliyoruz.
print("------------------------------")

# FİLTER
def filter_function(x):                    # filtre de kullanmak üzere fonksiyon tanımladık.
    return x["degisken1"].std() > 9

print(df.groupby("gruplar").filter(filter_function))    # grupladık, filtrenin içine fonksiyonumuzu yazdık.
# fonksiyona gidiyor, degisken1 standart sapması 9 dan büyük columnları getiriyor.
# Tamamının std'si görmek için ; df.groupby("gruplar").std()   yi kullanabilirsin.
print("------------------------------")

# TRANSFORM
# groupby veya groupby'sız yapılabilir.
print(df["degisken1"] * 9)      # Böyle işlemler yapabiliriz. fakat daha karmaşık birşey istiyorsak AŞAGIYA BAK.
print("------------------------------")
df_a = df.iloc[:,1:3]       # Aşağıda hata almamak için kategorik olan gruplar columndan kurtulduk.
print(df_a.transform(lambda x: x-x.mean() )) # Gruplar kategorik olduğu için sayısal işlem yapamıyor hata veriyor. o yüzden bir üstte gruplardan kurtuluyoruz.
print("------------------------------")
print(df_a.transform(lambda x: (x-x.mean() )/ x.std() )) # Yukarıdakine ek her birinin std'sine böldük. Standartlaştırdık
print("------------------------------")

# APPLY
# groupby veya groupby'sız yapılabilir.
# groupby'sız çalışacagız o yüzden grupları silelim.

df = pd.DataFrame({ 'degisken1' : [12,23,33,22,11,99],
                    'degisken2': [100,253,333,262,111,969]},
                    columns = ['degisken1','degisken2'])
print(df)
print("------------------------------")
print(df.apply(np.sum))                 # apply ile toplamlarını bulduk.
print("------------------------------")
print(df.apply(np.sum))                 # mean veya herhangibir kendi yazdıgımız fonksiyonu gezdirebiliriz.
print("------------------------------")

# Grup bazlı deneyelim.
df = pd.DataFrame({'gruplar': ['A','B','C','A','B','C'],
                    'degisken1' : [12,23,33,22,11,99],
                    'degisken2': [100,253,333,262,111,969]},
                    columns = ['gruplar','degisken1','degisken2'])

print(df.groupby("gruplar").apply(np.sum))  ## sonucu inceleyin. # n.sum yerine "mean", "median" ... istediğinizi yazabilirsiniz.


















