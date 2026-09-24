import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("../data/weather_data.csv")

features = ["humidity", "pressure", "wind_speed", "rainfall"]
X = df[features]
y = df["temperature"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

new_weather = pd.DataFrame({
    "humidity": [55],
    "pressure": [1013],
    "wind_speed": [6],
    "rainfall": [0]
})

prediction = model.predict(new_weather)
print("Predicted Temperature:", round(prediction[0], 2), "°C")
