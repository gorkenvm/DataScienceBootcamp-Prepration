import seaborn as sns
import pandas as pd
# GRUPLAMA İŞLEMLERİ
df = pd.DataFrame({'gruplar':['A','B','C','A','B','C'],
                   'veri': [10,11,52,23,43,55]}, columns=['gruplar','veri'])

pd.options.display.width = 0    # if u cannot see all columns, do some settings with that than u ll see all columns
# or u can write that 'to_string()' end of ur line. like that: df.describe().to_string()

print(df)
print("------------------------")
print(df.groupby('gruplar').mean()) # Gruplama yapınca Aggregation işlemi yapmak zorundayız.
print("------------------------")
df = sns.load_dataset("planets")
print(df.groupby("method")["orbital_period"].mean()) # method'u grupla, orbital_period'un ortalamasını yaz.
print("------------------------")
print(df.groupby("method")["orbital_period"].describe()) # method'u grupla, orbital_period'u describe et.
