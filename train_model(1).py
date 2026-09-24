import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("../data/weather_data.csv")

features = ["humidity", "pressure", "wind_speed", "rainfall"]
target = "temperature"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_prediction = linear_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)

def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    r2 = r2_score(actual, predicted)
    print("\n" + name)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

evaluate_model("Linear Regression", y_test, linear_prediction)
evaluate_model("Random Forest", y_test, rf_prediction)
