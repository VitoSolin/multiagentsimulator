# data_loader.py
import polars as pl

def load_avazu(path: str):
    """
    Baca Parquet Avazu, return Polars DataFrame tanpa kolom 'id'
    """
    df = pl.read_parquet(path).drop("id")
    # Ubah kolom kategorikal → integer (string tetap boleh, LightGBM akan handle)
    # di Polars tidak wajib di-encode dulu; tapi kalau mau:
    # df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in df.columns])
    return df
