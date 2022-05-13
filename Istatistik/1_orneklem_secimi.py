import numpy as np
# Bir şehirdeki kişilerin yaş ortalamasını incelemek istiyoruz fakat fiziksel olarak mümkün değildir. Örnekteki 10.000 kişi şehirdeki kişileri ifade ediyor.
# Tamamını inceleyemeyeceğimiz için 10 tane farklı 100 er kişi seçiyoruz. Populasyondan 10 tane örneklem çekiyoruz ve bunların ortalamasına bakıyoruz.
# İstatistik diyor ki bu örneklemin ortalaması popülasyonun ortalamasına çok yakın olacaktır. Bu yüzden de tüm şehrin yaşını kaydetmek yerine sadece rastgele örneklemler ile işi bitiriyoruz.

population = np.random.randint(0, 80, 10000) # created a random population serie
print(population[0:10])

# Sample Choosing
np.random.seed(115)                          # Hep aynı örneklemi çekmesini sağlar.
orneklem = np.random.choice(a = population, size = 100) # seçimi gerçekleştiriyoruz.
print(orneklem[0:10])

print(orneklem.mean())                      # both mean almost same
print(population.mean())

# Sample Distrubition
np.random.seed(10)
orneklem1 = np.random.choice(a = population, size = 100)
orneklem2 = np.random.choice(a = population, size = 100)
orneklem3 = np.random.choice(a = population, size = 100)
orneklem4 = np.random.choice(a = population, size = 100)
orneklem5 = np.random.choice(a = population, size = 100)
orneklem6 = np.random.choice(a = population, size = 100)
orneklem7 = np.random.choice(a = population, size = 100)
orneklem8 = np.random.choice(a = population, size = 100)
orneklem9 = np.random.choice(a = population, size = 100)
orneklem10 = np.random.choice(a = population, size = 100)
# samples mean is almost same of general mean
print((orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean() 
+ orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean() )  / 10)

print(orneklem1.mean()) # mean of orneklem1 and orneklem2 is not same
print(orneklem2.mean())









