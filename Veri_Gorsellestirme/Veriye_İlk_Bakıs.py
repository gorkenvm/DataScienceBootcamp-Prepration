import pandas as pd
pd.options.display.width = 0
import seaborn as sns

##### VERİ SETİ HİKAYESİ VE YAPISININ İNCELENMESİ ######

planets = sns.load_dataset("planets")
#print(planets.head())
# Veri Setinin Hikayesi Nedir ?
"""""
Nasanın yayınladığı galaksi keşfi ile ilgili bir veri setidir.
method = Gezegen/galaksilerin bulunma şeklini ifade eder.
number = Bulunan sistemlerdeki gezegen sayısını ifade etmektedir.
orbital_period = yörünge dönemi
mass = kütle
distance = uzaklık
year = bulunma tarihi

Not:**// Verinin kaydedilme şekli nedir ?
Birleştirme yapmamız gerekecektir. O yüzden bileşenlerinin oluşma hikayesinin ne olduğunu bilmek lazım.
Hangi değişkenin nasıl oluştuğunu bilmemiz gerekiyor.
"""
df = planets.copy()     # yedekleyip üzerinde çalışılmasını tavsiye ederim.

print(df.head(),'\n',df.tail()) # ilk ve son 5 veriyi alıp bir bakalım column ve rowlara
print("------------------------------------")
print(df.info())        # Veri setinin yapısal bilgilerini verir.
"""
* Pandas DataFrame'dir.
* 1.035 gözlem(row) ve 6 column(değişken) vardır.
* 3 sürekli sayısal (float), 2 kesikli sayısal(int), 1 kategorik(object) gibi düşünebilirsiniz veri tipi vardır.
* Hangilerinde null var görebiliyoruz.
"""
print("------------------------------------")
df.method = pd.Categorical(df.method)       # Kategorik olarak dönüştürdük.


##### VERİ SETİNİN BETİMLENMESİ ######

print(df.shape)
print("------------------------------------")
print(df.columns)
print("------------------------------------")
print(df.describe().T)             # Kategorik değişkenleri dışarıda bırakır ve eksik gözlemleri gözardı eder.
"""
Number: Ortalama 1.78 çok düşük, std 1.24 çok düşük, ortalamadan bile düşük, tuhaf geldi
baktıgımızda min 1 max 7 gezegen olduğunu gördük, yani mean ve std nin küçük olması gayet normal
Ayrıca dağılımın 1 e yakın oldugunu anlıyoruz. Bunla ilgili bir çıkarım yapamayacağız. Devam
orbital_period: mean 2.002 yüksek bir değer, std 26.014 çok çok yüksek bi değer.
min değeri 0.090 tuhaf görünüyor dikkatimi çekti. nasıl min 0.090 max'ı nasıl 730.000 gibi bir değer olabilir ki
demek ki dönemi ifade eden degerler bizim anladıgımız tarzda bilgi taşıyan değerler değilde büyüklük küçüklük ifade eden birşey olabilir.
mass: mean 2.6 bu bir birimdir, std 3.8 dir demekki galaksilerin kütleleri arasında çok ciddi fark yok.
min 0.0036 max 25, aha kütleleri yakın dedik ama hayır öyle değil min max arası çok uzak ve farklılar.
İlk column(number) da biliyoruz max 7 gezegenli galaksi var idi demek ki mass 25 olan bu galaksidir.
mean ve std rakamlarına bakarak büyük veya küçük diyemeyiz. bize kabaca merkezi ve etrafındaki dagılımı gösteriyor.
distance: mean 264 std 733, min 1.35 max 8.500 demekki bunda çok geniş bir yayılım var
year: Yıl'ı daha sonra yapacagız. 

"""

print("------------------------------------")










