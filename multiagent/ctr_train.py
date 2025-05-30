# ctr_train.py
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np

DATA = "avazu_dev_pro.parquet"
MODEL_OUT = "ctr_model_pro_two.txt"

# ---------- load ----------
df = pl.read_parquet(DATA).drop("id")  # buang kolom id

# ---------- feature engineering ----------
df = (
    df.with_columns([
        (pl.col("hour") % 100).alias("hour_of_day"),               # 0–23
        (((pl.col("hour") // 100) // 100) % 7).alias("day_of_week") # 0–6
    ])
)

# ... (bagian load & feature engineering tetap)

features = [
    "hour_of_day", "day_of_week", "banner_pos", "device_type", "device_conn_type",
    "site_category", "app_category", "C14", "C17", "C20", "C21"
]

X = df.select(features).to_pandas()
y = df["click"].to_numpy()

# ---- NEW: ubah kolom string → category dtype ----
cat_cols = ["site_category", "app_category"]
for col in cat_cols:
    X[col] = X[col].astype("category")

# ---------- split ----------
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- lightgbm ----------
import lightgbm as lgb
train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
val_set   = lgb.Dataset(X_val,   label=y_val,   categorical_feature=cat_cols)

params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.1,
    num_leaves=64,
    verbose=-1,
)

model = lgb.train(
    params,
    train_set,
    valid_sets=[val_set],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50)]     # versi callback
)

print("Best AUC:", model.best_score["valid_0"]["auc"])
model.save_model("ctr_model_pro_two.txt")
print("✅ Model saved to ctr_model_pro_two.txt")
