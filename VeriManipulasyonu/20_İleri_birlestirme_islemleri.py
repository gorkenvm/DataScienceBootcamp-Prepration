import pandas as pd

# Birebir birlestirme
df1 = pd.DataFrame({'calisanlar':['Ali','Veli','Ayse','Fatma'], 'grup':['Muhasebe','Muhendislik', 'Muhendislik', 'İK']})
print(df1)
print("-----------------------------------------")
df2 = pd.DataFrame({'calisanlar': ['Ayse','Ali','Veli','Fatma'], 'ilk_giris': [2010,2009,2014,2019]})
print(df2)
print("-----------------------------------------")
print(pd.merge(df1,df2))            # Calisanlar ortak oldugu için calışanlar uzerınde bırlestırdı.
print("-----------------------------------------")

# Çoktan Tek'e   -   Many to One

df3 = pd.merge(df1,df2)         # df1 ve df2 birleştirildi.
print(df3)
print("-----------------------------------------")
df4 = pd.DataFrame({'grup': ['Muhasebe','Muhendislik','İK'], 'mudur':['Caner','Mustafa','Berkcan']})
print(df4)
print("-----------------------------------------")
print(pd.merge(df3,df4))
print("-----------------------------------------")
## Many to Many
df5 = pd.DataFrame({'grup': ['Muhasebe','Muhasebe','Muhendislik','Muhendislik','İK','İK'],
                    'yetenekler':['matematik','excel','kodlama','linux','excel','yonetim']})
print(df5)
print("-----------------------------------------")
print(df1)
print("-----------------------------------------")
print(pd.merge(df1,df5))        # df5 çok df1 tek, tüm çalışanlar çokladı.









