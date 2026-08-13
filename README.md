# Academic-weapon
A team based on 4 people: Hani Habib, Danny Zheng, Valdemar RIng og Thor Carlsen
import pandas as pd

gini = pd.read_excel("GINI.xlsx")
deciles = pd.read_excel("deciler.xlsx")

print(gini.head())
print(deciles.head())

# Årene står i række 1, og Gini-værdierne står i række 2
gini_table = pd.DataFrame({
    "År": gini.iloc[1, 1:].values,
    "Gini": gini.iloc[2, 1:].values
})

# Fjern tomme værdier
gini_table = gini_table.dropna()

print(gini_table)

import matplotlib.pyplot as plt

plt.plot(gini_table["År"], gini_table["Gini"])
plt.title("Gini-koefficient over tid")
plt.xlabel("År")
plt.ylabel("Gini-koefficient")
plt.show()

