
import seaborn as sns
tips = sns.load_dataset("tips")  # this data set is an library in seaborn
df = tips.copy()
print(df.head())

print(df.describe().T)  # Describe shows you some statistics about your columns value. T is transpose, You can see result better by .T

# İf u dont have this library, Go AppData/Local/Python/Python39/Lib   (this is my path where my python libs are located). When u r here at Git Bash then write that pip install researchpy
import researchpy as rp

print(rp.summary_cont(df[["total_bill","tip","size"]])) # Sayısal veriler için

print(rp.summary_cat(df[["sex","smoker","day"]]))       # kategorik veriler

print(df[["tip","total_bill"]].cov())                   # Değişkenlerin ilişkisi arasında ki fark

print(df[["tip","total_bill"]].corr())                  # Değişkenleri arasındaki ilişki

