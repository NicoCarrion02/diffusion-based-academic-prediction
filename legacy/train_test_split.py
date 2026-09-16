import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(pd.read_parquet('data/pisa2022.parquet'), test_size=test_size, random_state=random_state)
    print(f"Train rows: {len(train_df)} ({(1-test_size)*100:.1f}%)")
    print(f"Test rows: {len(test_df)} ({test_size*100:.1f}%)")
    return train_df, test_df

def save_split(train_df, test_df):
    train_df.to_parquet('data/train.parquet', index=False)
    print("Saved train split to data/train.parquet")

    test_df.to_parquet('data/test.parquet', index=False)
    print("Saved test split to data/test.parquet")

if __name__ == "__main__":
    train_df, test_df = split_data()
    save_split(train_df, test_df)
