# Tamamıyla Rastgele oluşan bir eksik veri değilse silmek yada doldurmak
# daha ciddi yapısal sorunlara neden olabilir o yüzden rastsallıpı incelemek için
# Görselleştirme yapacağız.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

V1 = np.array([1, 3, 6, np.NaN,7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN,5,8,12,np.NaN, np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
df = pd.DataFrame(
    {"V1" : V1,
     "V2" : V2,
     "V3" : V3}
)
print(df)

import missingno as msno    # Görselleştirmek için kullanacağız.

msno.bar(df)
#plt.show()  # Solda yüzde, üstte eksik olmayan veri, sağda toplam ve var olan veri sayısı yazar.
msno.matrix(df)
#plt.show()  # Sağ tarafta dolu ve boş veri sayısını veriyor.
# Daha geniş bir veri seti üzerinden inceleyelim
import seaborn as sns
df = sns.load_dataset('planets')
df.head()
print(df.isnull().sum())    # Çokça eksik veri var,
# Rastsallığını inceleyelim.
msno.matrix(df)
plt.show()  # açık bir şekilde görünüyor ki, orbital peroid'ta olan eksik veri var ise, mass'te karşılığıda
# eksiktir. Tamamı için böyle, o yüzden kesinlikle bir bağımlılık var. Rastsallık gözlemlenemiyor.
# distance ve mass' i karşılaştırdıgımız da aynı şeyi söyleyemiyoruz. karşılıklı değerleri dolu da olabiliyor bazen.
msno.heatmap(df);
plt.show()  # Burada korelasyona bakıyoruz. 1 ise tamamen ilişkili, 0 ise aralarında ilişki yoktur diyebiliriz.
# orbital_period ve mass arasında 0.2, mass ve distance arasıdna 0.5 korelasyon oldugunu numerik olarak grafik bize söylüyor.
# Tamamını ele alınca aralarında yeteri kadar kolerasyon olmadığı için, Rastsallıktan bahsedemeyiz.
# Bu durumda direk silme ve doldurma işlemi yapmak beraberinde yapısal problemler oluşturacaktır ve tavsiye edilmez hatta yapmayın denir.








