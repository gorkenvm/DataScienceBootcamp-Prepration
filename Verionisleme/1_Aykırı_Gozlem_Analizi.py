import seaborn as sns
import matplotlib.pyplot as plt

################# TEK DEĞİŞKENLİ AYKIRI GÖZLEM ANALİZİ #########################

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64']) # Sadece sayısal verilere ihtiyacımız var
df = df.dropna()
print(df.head())
# 'table için aykırı gözlem analizi yapacağız.
df_table = df['table']
print(df_table.head())
#sns.boxplot(x = df_table);
#plt.show()

Q1 = df_table.quantile(0.25)
Q3 = df_table.quantile(0.75)
print('Q1= %.1f, Q3= %.1f' % (Q1,Q3))   # Q1 ve Q3
IQR = Q3 - Q1                           # Q3 ve Q1 farkı IQR bulundu
print('IQR=',IQR)
alt_sinir = Q1 - 1.5 * IQR              # Alt ve üst sınır bulundu ve atandı.
ust_sinir = Q3 + 1.5 * IQR
print('Alt Sınır= %.f, Üst Sınır= %.f' % (alt_sinir, ust_sinir))
print((df_table < alt_sinir) | (df_table > ust_sinir)) # alt sınırdan küçük, üst sınırdan büyük olanları sorguladık. True olanlar arkırı değerlerimizdir.
# Takip edebilmek adına alt sınırı inceleyelim ve fancy ile değerlere ulaşalım.
aykiri_tf = df_table < alt_sinir
print(df_table[aykiri_tf])           # Fancy ile verilere ulaştık
print(df_table[aykiri_tf].index)     # İşlem yapabilmek için index'lerine de ulaştık.


#### AYKIRI DEĞERLERİ YUKARIDA YAKALADIK
#### AYKIRI DEĞER PROBLEMİNİ ÇÖZECEĞİZ

# SİLME

import pandas as pd
print(type(df_table)) # Seri olarak görüntülüyoruz. İşlem yapabilmek adına DataFrame'e çevireceğiz.
df_table = pd.DataFrame(df_table)
print(df_table.shape)
# Aykırı olmayanlara ulaşmaya calışacağız. t_df temiz df olarak yazıyorum.
t_df = df_table[~( (df_table < alt_sinir) | (df_table > ust_sinir) ).any(axis= 1)]
print(t_df)
print(t_df.shape)

# ORTALAMA İLE DOLDURMA
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
print(df.head())
df_table = df["table"]      # Veriyi baştan oluşturduk ki silinenler geri gelsin.
print(aykiri_tf.head())     # aykiri_tf de değişiklik yok, bunu kullanabiliriz.
print(df_table.mean())      # Ortalamayı bulduk atamak için
df_table[aykiri_tf] = df_table.mean()   # aykırı değerleri ortalama ile doldurduk.
print(df_table[aykiri_tf])              # Gözlemlendiği üzere tamamı ortalama ile dolduruldu.
## Aykırı değerleri silmek istemedigimizde kullanabileceğimiz bir yöntemdir.

# BASKILAMA YÖNTEMİ
## Aykırı değerler üst tarafta ise üst sınıra eşitlenir, alt tarafta ise alt sınıra eşitlenir.
## Böylece aykırı değerlerin veriyi yukarı veya aşağı çekme çabası göz önünde bulunurulmuş olup
## Bazı senaryolarda ortalama ile doldurmaktan çok daha mantıklı olacaktır.
## Ortalama ile sınırlar arası mesafe fazla ise sınırlara taşıyarak törpülemiş olmak daha mantıklı olur.
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
print(df.head())
df_table = df["table"]      # Veriyi baştan oluşturduk ki silinenler geri gelsin.
print(df_table[aykiri_tf])  # aykırılara ulaştık.
print(alt_sinir)            # alt sınır'ı hatırlayalım
df_table[aykiri_tf] = alt_sinir
print(df_table[aykiri_tf])      # gözlemlendiği üzere alt sınıra eşitlemiş olduk

# Aynı işlemi üst sınıf içinde yapalım

aykiri_ft = df_table > ust_sinir    # üst sınırdan büyük olanları yakalık.
print(df_table[aykiri_ft])          # üst aykırı değerleri gözlemleyelim
df_table[aykiri_ft] = ust_sinir     # üst sınıra eşitledik
print(df_table[aykiri_ft])          # Gözlemlendiği gibi üst aykırı değerlerde üst sınıra baskılanmış oldu.



################# ÇOK DEĞİŞKENLİ AYKIRI GÖZLEM ANALİZİ #########################
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64']) # Sadece sayısal verilere ihtiyacımız var
df = df.dropna()
print(df.head())

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors= 20, contamination= 0.1)   # n_neighbors = komşuluk sayısı, contamination = yoğunluk
clf.fit_predict(df) # modeli tahmin etmesi için fit ediyoruz.
df_scores = clf.negative_outlier_factor_    # atama işlemi ile kaydettik.
print(np.sort(df_scores)[:20])              # sort edip ilk 20 gözlemi inceliyoruz.
# Gözlem yaptıktan sonra eşik değer kabul etmemiz lazım, burada eşik değeri rastgele belirliyoruz.
esik_deger = np.sort(df_scores)[13]     # Esik deger olarak atama yaptık.
aykiri_tf = df_scores > esik_deger      # aykırı olmayanlar için true'ları çektik

### SİLME İŞLEMİ

yeni_df = df[df_scores > esik_deger]    # yeni_df olarak aykırı olmayanları aldık yani aykırıları silmiş olduk.
print(yeni_df)
aykirilar = df[~aykiri_tf]
print(aykirilar)       # Bu da aykırı gözlemler.

### BASKILAMA İŞLEMİ

#print(df[df_scores == esik_deger])  # Eşik değer belirlemiştik ve bu değere karşılık gelen gözlem 32230 index'li gözlemdir.
# Baskılayacağımız aykırı değerlere bu eşik değerlerini atayacağız.
baskı_deger = df[df_scores == esik_deger]   # Baskı değerimiz
aykirilar = df[~aykiri_tf]      # aykirilar, silme bölümünde bulmuştuk.
# Aykırıları baskı deger'e eşitleyeceğiz fakat index problemi olacak bu yüzden indexlerden kurtulmamız lazım.
# indexlerden kurtulmak için array'a dönüştürüyoruz.
res = aykirilar.to_records(index= False)    # aykırıları array'e çevirdik ve indexsiz olarak kaydettik
res[:] = baskı_deger.to_records(index= False) # Baskı değerlerini de index'siz array'e dönüştürdük ve res'in tüm elemanlarına atadık.
print(res)      # Buradan daha net gözlemleyebiliriz.
print(df[~aykiri_tf])   # gerçek veri setinde değişiklik yok, şimdi de res array'ini bu veri setinin içine koyacağız.

df[~aykiri_tf] = pd.DataFrame(res, index = df[~aykiri_tf].index) # orjinal veride bulunan aykırı degerlere atama yapıyuoruz.
# dataframe olarak res'i atayacagız. İndex'lerinide, df[~aykiri_tf]'nin indexleri olarak ayarlıyoruz.
print(df[~aykiri_tf])   # Buradan orjinal verideki aykırı degerlerin baskılandığını gözlemleyebiliriz.






