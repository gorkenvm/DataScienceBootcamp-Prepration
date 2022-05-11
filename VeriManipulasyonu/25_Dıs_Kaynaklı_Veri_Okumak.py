import pandas as pd

df = pd.read_csv("reading_data/ornekcsv.csv", sep= ";")
# sep ön tanımlı virgül ve boşluktur. bu dosyada noktalıvirgül ile ayrıldıgı için sep kullanmak zorundayız.
print(df)
print("----------------------------------------")
#txt okuma
df2 = pd.read_csv("reading_data/duz_metin.txt")
print(df2)
print("----------------------------------------")
df3 = pd.read_excel("reading_data/ornekx.xlsx")
print("----------------------------------------")
print(type(df),'\n', type(df2),'\n', type(df3)) # OO hepsi DataFrame ozmn tüm pandas özelliklerini kullanabliriz.
print("----------------------------------------")
print(df.head())
print("----------------------------------------")
df.columns = ("A","B","C")
print(df.head())













