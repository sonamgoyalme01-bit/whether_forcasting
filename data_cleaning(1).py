import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates()
    df = df.dropna()
    return df

if __name__ == "__main__":
    df = load_data("../data/weather_data.csv")
    df = clean_data(df)
    print(df.head())
    print(df.info())
