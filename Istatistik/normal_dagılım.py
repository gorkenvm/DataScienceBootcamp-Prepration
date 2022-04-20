# İş Uygulaması: Ürün satış olasılıklarının hesaplanması.
# Problem: Gelecek ay ile ilgili satışların belirli değerlerde gerçekleşmesi olasılıkları belirlenmek isteniyor.
# Dağılımın normal olduğu biliniyor. Görselleştirme veya hipotez testi ile bilinebilir.
# aylık mean = 80K, SD = 5K

# Soru 90K dan fazla satış yapma olasılığı nedir ?

from scipy.stats import norm
# Satışların 90K dan fazla olma olasılığı
# 1 den çıkarınca yukarısı, yani 90dan fazla olanlar. 1 den çıkarmadan yaparsak 90dan az onaları verir.
print(1- norm.cdf(90, 80, 5)) # hesaplanmak istenen, mean, std
# 70K dan fazla olma olasılığı
print(1- norm.cdf(70, 80, 5))
# 73K dan az olma olasılığı
print(norm.cdf(73, 80, 5))
# 85-90K arasında olma olasılığı
print(norm.cdf(90, 80, 5) - norm.cdf(85, 80, 5))

