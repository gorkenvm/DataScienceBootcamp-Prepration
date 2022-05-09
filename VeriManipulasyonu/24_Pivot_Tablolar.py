import pandas as pd
pd.options.display.width = 0 # Yine columnların tamamını göstermesi için seçenekler->görüntüle->genişlik = 0 dedik.
import seaborn as sns
# Grupby ile karıştırılabiliyor. Grupby'ın çok boyutlu versiyonu olarak düşünülebilir.
titanic = sns.load_dataset('titanic')
print(titanic.head())
print("---------------------")
print(titanic.groupby('sex')["survived"].mean()) # cinsiyete göre grupladık. survived ortalamasını aldık.
print("---------------------")
print(titanic.groupby('sex')[["survived"]].mean()) # "survived"'a [] köşeli parantez ekledik df oldu. daha güzel görünüyor. Aslında pivot işlemi yaptık.
print("---------------------")
print(titanic.groupby(["sex","class"])[["survived"]].aggregate("mean"))
print("---------------------")
print(titanic.groupby(["sex","class"])[["survived"]].aggregate("mean").unstack()) # unstack() ile daha farklı görüntüleyebilirsin. Hangisi okuması kolaysa onu yap.
print("---------------------")

# Pivot ile pivot table
# Yukarıdaki işlemlerin kolay yollarını pivot ile yapacağız.

print(titanic.pivot_table("survived", index="sex",columns="class")) # pivot_table(odaklandığın değişken, neyi gruplayacaksın, daha da grupla)
print("---------------------")
# age yani yaş column'ı kategorik'e çevirelim.
age = pd.cut(titanic["age"], [0, 18, 90]) # pd.kes(neyi keseyim, nasıl keseyim aralık verirsen daha iyi olur)
print(age.head(10))
print("---------------------")
print(titanic.pivot_table("survived", ["sex", age], "class")) # yukarıda ki gibi index column yazmadık, Neden?
# sırasını doğru biliyorsan onları yazmana gerek yok. sırasından emin değilsen index 2. sırada column 3. sırada
# ozmn index ve column'u da yaz.
# Farklı olarak yaşları da kategorik veriye çevirdik













