import pandas as pd

def extract_data():
    train_df = pd.read_parquet('data/train.parquet')
    test_df = pd.read_parquet('data/test.parquet')
    return {'train': train_df, 'test': test_df}

def transform_data(df):
    math_cols = [f'PV{i}MATH' for i in range(1, 11)]
    available = [c for c in math_cols if c in df.columns]
    if available:
        df['PV_MATH_MEAN'] = df[available].mean(axis=1)
        print(f"Transformed {len(available)} PV_MATH columns into PV_MATH_MEAN.")
    else:
        print("No PV_MATH columns found for transformation.")
    
    pv_columns = [col for col in df.columns if col.startswith('PV')]
    pv_columns.remove('PV_MATH_MEAN')  # Keep the target column
    df.drop(columns=pv_columns, inplace=True)
    
    return df

def load_data(df, file_name):
    df.to_parquet(file_name, index=False)
    print(f"Saved transformed data to {file_name}")

if __name__ == "__main__":
    for name, df in extract_data().items():
        df = transform_data(df)
        load_data(df, file_name=f"data/{name}_transformed.parquet")
