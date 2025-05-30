import pandas as pd

# Step 1: Baca sebagian saja (200 ribu baris pertama)
print("Loading CSV sample...")
df = pd.read_csv("train.csv", nrows=1_000_000)

# (Opsional: randomize sampel biar nggak cuma ambil header)
# df = df.sample(n=200_000, random_state=42)

# Step 2: Simpan ke Parquet agar lebih cepat & ringan
print("Saving as Parquet...")
df.to_parquet("avazu_dev_pro.parquet", index=False, compression="zstd")

print("Done! ✅ Saved as avazu_dev.parquet")
