import pandas as pd

def sas_to_parquet():
    df = pd.read_sas(
        "data/CY08MSP_STU_QQQ.SAS7BDAT",
        format="sas7bdat"
    )
    print("Loaded SAS file into DataFrame")

    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    df.to_parquet("data/pisa2022.parquet", index=False)
    print("Converted SAS file to Parquet format and saved as data/pisa2022.parquet")

if __name__ == "__main__":
    sas_to_parquet()