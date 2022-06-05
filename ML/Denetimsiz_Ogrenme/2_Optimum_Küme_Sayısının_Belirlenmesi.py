# İş bilgisi ile zaten segment sayısını biliyor olacağız.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# VERİ SETİ
df = pd.read_csv(r"C:\Users\PC\Desktop\VeriBilimiOkulu\MakineOgrenmesi\datasets\USArrests.csv", index_col= 0)

# ELBOW YÖNTEMİ
ssd = []
K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters= k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel('Farklı K Değerlerine Karşılık  Uzaklık Artık Toplamları')
plt.title('Elbow Yöntemi')
plt.show()      # eğimin keskin değiştiği yer optimum n_clusters olacaktır.

# Alternatif yöntem. Bu yöntemi çok daha fazla sevdim ayrıntılı. k 6 olmalıdır diyor.
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k = (2,20))
visu.fit(df)
visu.poof()
plt.show()

kmeans = KMeans(n_clusters= 4).fit(df)
kumeler = kmeans.labels_
pd.DataFrame({'Eyaletler': df.index, 'Kumeler': kumeler})
df['Kume_No'] = kumeler
print(df)