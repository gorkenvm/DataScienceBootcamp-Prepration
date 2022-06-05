# Temel fikir, Çok değişkenli verinin ana özelliklerini daha az sayıda değişken/bileşen ile temsil etmektir.
# Daha çok görüntü işleme ve regresyon problemlerinde kullanılıyor.
# Değişkenler bir birine benziyorsa çoklu değişken problemi çıkıyor. indirgeme işleminen sonra korelasyon kalmıyor.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# VERİ SETİ
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\Hitters.csv")
df.dropna(inplace=True)
df = df._get_numeric_data() # Sadece numerik sayıları seç
print(df.head())

# Amaç gözlem sayısını 2 3 e düşürmek
from sklearn.preprocessing import StandardScaler
df = StandardScaler().fit_transform(df)
print(df[:5,:5])
from sklearn.decomposition import PCA   # Temel bileşen indirgemesi yapacağız
pca = PCA(n_components= 2)
pca_fit = pca.fit_transform(df)
bilesen_df = pd.DataFrame(data= pca_fit, columns=['Birinci', 'İkinci'])
print(bilesen_df)

print(pca.explained_variance_ratio_)    # %70 e yakın ilişki halen elimizde
print(pca.components_)
# Optimum bileşen sayısı
pca = PCA().fit(df)
plt.plot(np.cumsum((pca.explained_variance_ratio_)))
plt.show()  # 3 bileşenle %80 bilgi taşıyabiliyoruz

# FİNAL
pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)
print(pca.explained_variance_ratio_)









