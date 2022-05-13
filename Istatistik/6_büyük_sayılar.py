# BÜYÜK SAYILAR YASASI
# Bir rassal değişkenin uzun vadeli kararlılığını tanımlayan olasılık teoremidir.
# Yazı tura deneyi fazla sayıda denediğimze hep %50 aralığında çıkar.
import numpy as np

rng = np.random.RandomState(123)  # RandomState, random işlemini sabitler.
rng.randint(0,2, size = 5)

for i in np.arange(1,21):
    deney_sayisi = 2**i
    yazi_turalar = rng.randint(0,2, size = deney_sayisi)
    yazi_olasiliklari = np.mean(yazi_turalar)
    print("Atış Sayısı:", deney_sayisi, "---", 'Yazı Olasılığı: %.2f' % (yazi_olasiliklari * 100))  # .2 virgülden sonra 2 rakamı yazdır demek.
