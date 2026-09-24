import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/weather_data.csv")
df["date"] = pd.to_datetime(df["date"])

print(df.head())
print(df.shape)
print(df.describe())

plt.figure(figsize=(10,5))
plt.plot(df["date"], df["temperature"])
plt.title("Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["humidity"], kde=True)
plt.title("Humidity Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x=df["humidity"], y=df["temperature"])
plt.title("Temperature vs Humidity")
plt.xlabel("Humidity (%)")
plt.ylabel("Temperature (°C)")
plt.show()

plt.figure(figsize=(8,6))
corr = df[["temperature","humidity","pressure","wind_speed","rainfall"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Weather Feature Correlation")
plt.show()
