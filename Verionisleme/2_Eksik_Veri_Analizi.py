# EKSİK VERİ ANALİZİ

import numpy as np
import pandas as pd

V1 = np.array([1, 3, 6, np.NaN,7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN,5,8,12,np.NaN, np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
df = pd.DataFrame(
    {"V1" : V1,
     "V2" : V2,
     "V3" : V3}
)
print(df)
# HIZLI ÇÖZÜM
print(df.isnull().sum())    # eksik verilerin toplamına ulaşıyoruz.
print(df.notnull().sum())   # eksik olmayan veriler
print(df.isnull())          # TF sorgusu yaptık, df nin içine koyunca bize getirecek verileri.
print(df[df.isnull().any(axis= 1)])     # En az 1 veri eksik olanları getir.
print(df[df.notnull().all(axis= 1)])    # Hepsi dolu olanları getirç.

# Eksik Değerlerin Direk Silinmesi
print(df.dropna())          # 1 tane dahi eksik ver var ise o gözlemleri sil.
# kalıcı olarak silmesi için dropna(inplace = True) yazmamız laızm.

# Basit Değer Atama
print(df["V1"].fillna(df["V1"].mean()))     # V1 içindeki NaN'leri df V1'in ortalaması ile d0oldur.
#####df["V2"] = df["V2"].fillna(0) # Bu şekilde yazınca kaydediyor, biz sadece görüntülüyoruz.
print(df["V2"].fillna(0))                   # V1 içindeki NaN'leri 0 ile doldur.

# Tüm df yi fonksiyonel programlama ile dolduralım
print(df.apply(lambda  x: x.fillna(x.mean()), axis= 0))    # apply = tüm sütunara uygula
# neyi uygulayacağım, lambda fonksiyonu ile veriyorum. x'e bağlı bir değişken var.
# x NaN ise doldur. ne ile doldurayım? x'in sütun ortalaması ile doldur.













