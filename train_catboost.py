"""Train the 2024 JoSAA Cutoff CatBoost rank estimator.

Install once:  pip install -r requirements.txt
Run from this directory: python train_catboost.py
"""
from pathlib import Path
import json
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "josaa_2024_round5.csv"
MODEL = ROOT / "josaa_2024_catboost_rank_estimator.cbm"

df = pd.read_csv(DATA)
target = "Closing Rank"
features = ["Institute", "Academic Program Name", "Quota", "Seat Type", "Gender", "Opening Rank"]
df = df[features + [target]].dropna().copy()
for column in features[:-1]:
    df[column] = df[column].astype(str)
df["Opening Rank"] = pd.to_numeric(df["Opening Rank"], errors="coerce")
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna()

# Log target makes the wide AIR scale more learnable while retaining rank ordering.
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target].clip(lower=1).map(__import__("math").log1p),
    test_size=0.20, random_state=42,
)
cat_features = [i for i, feature in enumerate(features) if feature != "Opening Rank"]
model = CatBoostRegressor(
    loss_function="RMSE", iterations=650, depth=8, learning_rate=0.05,
    l2_leaf_reg=5, random_seed=42, verbose=False,
)
model.fit(X_train, y_train, cat_features=cat_features)
predicted_rank = model.predict(X_test)
actual_rank = y_test
metrics = {
    "rows": len(df),
    "target": target,
    "mae_log_rank": round(float(mean_absolute_error(actual_rank, predicted_rank)), 4),
    "r2_log_rank": round(float(r2_score(actual_rank, predicted_rank)), 4),
    "features": features,
    "notes": "Model predicts log(1 + Round-5 closing AIR). Convert outputs with expm1().",
}
model.save_model(MODEL)
(ROOT / "catboost_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
print(f"Saved model: {MODEL.name}")
