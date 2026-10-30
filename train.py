"""
Credit Card Default Prediction using PyCaret (AutoML)
Group 8: Syed Fazal Mehdi (S011), Manas Dabhade (S025), Deven Bhole (S027)

Dataset: UCI "Default of Credit Card Clients" (Taiwan, 2005)
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Run:  python train.py
"""
import os, time, json
T0 = time.time(); TIMES = {}
import pandas as pd
from pycaret.classification import (
    setup, compare_models, tune_model, finalize_model, predict_model,
    save_model, pull, plot_model,
)

os.makedirs("outputs", exist_ok=True)
DATA = "data/credit.csv"

# 1. Load data (local copy; fall back to PyCaret's dataset repo)
if os.path.exists(DATA):
    df = pd.read_csv(DATA)
else:
    from pycaret.datasets import get_data
    df = get_data("credit")
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA, index=False)

print(f"Data shape: {df.shape}")
print("Default rate:", round(df["default"].mean() * 100, 2), "%")

# Hold out 5% as unseen data to simulate new customers
unseen = df.sample(frac=0.05, random_state=42)
data = df.drop(unseen.index)
unseen.to_csv("outputs/unseen_sample.csv", index=False)

# 2. Setup: preprocessing pipeline (train/test split, imputation,
#    categorical encoding, normalisation, class-imbalance fix)
setup(
    data=data,
    target="default",
    categorical_features=["SEX", "EDUCATION", "MARRIAGE"],
    normalize=True,
    fix_imbalance=True,          # SMOTE on training folds
    train_size=0.8,
    fold=5,
    session_id=42,
    verbose=False,
)

TIMES["setup"] = time.time() - T0; t = time.time()
# 3. Compare models (cross-validated leaderboard)
top3 = compare_models(n_select=3, sort="AUC",
                      include=["lr", "dt", "rf", "et", "lightgbm", "gbc", "ada", "nb", "knn", "ridge"])
leaderboard = pull()
leaderboard.to_csv("outputs/leaderboard.csv")
print("\n=== Model leaderboard ===\n", leaderboard)

TIMES["compare"] = time.time() - t; t = time.time()
# 4. Tune the best model
best = top3[0]
tuned = tune_model(best, optimize="AUC", n_iter=10, choose_better=True)
pull().to_csv("outputs/tuned_cv.csv")

TIMES["tune"] = time.time() - t; t = time.time()
# 5. Evaluate on hold-out test set
predict_model(tuned)
holdout = pull()
holdout.to_csv("outputs/holdout_metrics.csv")
print("\n=== Hold-out metrics ===\n", holdout)

for p in ["auc", "confusion_matrix", "feature"]:
    try:
        plot_model(tuned, plot=p, save="outputs")
    except Exception as e:
        print(f"plot {p} skipped: {e}")

TIMES["evaluate"] = time.time() - t; t = time.time()
# 6. Finalise on full data and save pipeline
final = finalize_model(tuned)
save_model(final, "outputs/credit_default_model")

# 7. Score unseen customers
preds = predict_model(final, data=unseen.drop(columns="default"))
preds["actual"] = unseen["default"].values
acc = (preds["prediction_label"] == preds["actual"]).mean()
print(f"\nAccuracy on unseen sample ({len(preds)} rows): {acc:.3f}")
preds.head(20).to_csv("outputs/unseen_predictions.csv", index=False)
TIMES["finalize_predict"] = time.time() - t
TIMES["total"] = time.time() - T0
json.dump(TIMES, open("outputs/pycaret_times.json", "w"), indent=1)
print(TIMES)
print("Done. Model saved to outputs/credit_default_model.pkl")
