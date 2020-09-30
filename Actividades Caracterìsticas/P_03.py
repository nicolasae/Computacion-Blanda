import requests
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data_url = "https://raw.githubusercontent.com/holtzy/data_to_viz/master/Example_dataset/4_ThreeNum.csv"

r = requests.get(data_url)

with open("gdp-life.txt", "w") as f:
    f.write(r.text)

df = pd.read_csv("gdp-life.txt")
print(df.head())

print("___")
print("The correlation is: ", np.corrcoef(df["gdpPercap"], df["lifeExp"])[0, 1])
print("___")

sns.lmplot("gdpPercap", "lifeExp", df).set_axis_labels(
    "Life expectancy", "GDP per capita"
)

plt.title("People live longer in richer countries")
plt.tight_layout()
plt.show()
