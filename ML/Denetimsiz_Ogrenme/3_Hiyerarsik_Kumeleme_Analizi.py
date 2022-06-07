# Amaç gözlemleri birbirlerine olan benzerliklerine göre alt kümelere ayırmaktır.
# Gözlemler daha fazla alt kümeye ayırılmak isteniyorsa bu yöntem kullanılabilir.
# K mean yönteminde sadece 3 yada 4 kümeye ayırabiliyorduk burada ise o 3 kümenin altında da kümelere ayırabiliriz.
# Birleştirici ve Bölümleyici diye ikiye ayrılır.
# Birleştirici: Gözlem sayısı kadar küme var
# Adım 1 : Bir birine en yakın olan iki gözlem bulunur
# Adım 2 : Bu iki nokta birleştirilerek yeni gözlem oluşturulur.
# Adım 3 : Aynı işlem tekrarlanarak yukarıya doğru çıkılır ve tek bir kümede toplanır. uzaklık kullanılarak belirlenir.
# Bölümleyici : 1 tane küme vardır.
# Adım 1 : iki alt kümeye ayrılır.
# Adım 2 : Gözlem sayısı kadar küme elde edilince ye kadar bu işlemler devam eder.

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# VERİ SETİ
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\USArrests.csv", index_col= 0)

hc_complete = linkage(df, 'complete')   # dendogram uyguluyoruz.
hc_avarage = linkage(df, 'average')

plt.figure(figsize=(10,5))
plt.title('Hiyerarşik Kümeleme Dendogramı')
plt.xlabel('Gözlem Birimleri')
plt.ylabel('Uzaklıklar')
dendrogram(hc_complete, leaf_font_size=10);      # leaf_font_size x ekseninde olacak yazıların boyutu
plt.show()

# 4 tane küme olsun diye karar verdik. hangi kümede ne kadar elamam var ?
plt.figure(figsize=(10,5))
plt.title('Hiyerarşik Kümeleme Dendogramı')
plt.xlabel('Gözlem Birimleri')
plt.ylabel('Uzaklıklar')
dendrogram(hc_complete, leaf_font_size=10, truncate_mode='lastp', show_contracted=True, p=4);      # leaf_font_size x ekseninde olacak yazıların boyutu
plt.show()

# Diğer yöntem
plt.figure(figsize=(10,5))
plt.title('Hiyerarşik Kümeleme Dendogramı')
plt.xlabel('Gözlem Birimleri')
plt.ylabel('Uzaklıklar')
dendrogram(hc_avarage, leaf_font_size=10, truncate_mode='lastp', show_contracted=True, p=4);      # leaf_font_size x ekseninde olacak yazıların boyutu
plt.show()











