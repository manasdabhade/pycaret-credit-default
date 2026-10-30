# %% [markdown]
# # Traditional ML workflow (scikit-learn) — baseline for comparison with PyCaret
# Group 8 · Same dataset, same split ratios, same 10 algorithms, same 5-fold CV, same SMOTE.
# Designed to run on Google Colab (upload data/credit.csv or let it download).

# %%
# !pip install -q scikit-learn imbalanced-learn lightgbm   # (Colab: lightgbm/imblearn usually preinstalled)
import time, os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score, precision_score,
                             f1_score, cohen_kappa_score, matthews_corrcoef, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

T0 = time.time()
SEED = 42
URL = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/credit.csv"
path = "data/credit.csv" if os.path.exists("data/credit.csv") else ("../data/credit.csv" if os.path.exists("../data/credit.csv") else URL)
df = pd.read_csv(path)
print(df.shape, "default rate:", round(df["default"].mean(), 4))

# %%
# Same protocol as PyCaret run: 5% unseen, then 80/20 stratified split
unseen = df.sample(frac=0.05, random_state=SEED)
data = df.drop(unseen.index)
X, y = data.drop(columns="default"), data["default"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8, stratify=y, random_state=SEED)

cat = ["SEX", "EDUCATION", "MARRIAGE"]
num = [c for c in X.columns if c not in cat]
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", StandardScaler(), num)])

def make_pipe(model):
    return Pipeline([("pre", pre), ("smote", SMOTE(random_state=SEED)), ("model", model)])

# %%
models = {
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "LightGBM": LGBMClassifier(random_state=SEED, verbose=-1),
    "AdaBoost": AdaBoostClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Ridge": RidgeClassifier(random_state=SEED),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"Accuracy": "accuracy", "AUC": "roc_auc", "Recall": "recall",
           "Prec.": "precision", "F1": "f1"}
rows = []
t_cmp = time.time()
for name, m in models.items():
    t = time.time()
    r = cross_validate(make_pipe(m), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
    rows.append({"Model": name, **{k: r[f"test_{k}"].mean() for k in scoring}, "TT (Sec)": (time.time() - t) / 5})
    print(f"{name:20s} AUC={rows[-1]['AUC']:.4f}")
compare_time = time.time() - t_cmp
lb = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
print(lb.round(4))

# %%
best_name = lb.iloc[0]["Model"]
best = make_pipe(models[best_name])
grid = {"model__n_estimators": [50, 100, 200, 300], "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [2, 3, 4, 5, 6], "model__subsample": [0.6, 0.8, 1.0]}
t_tune = time.time()
search = RandomizedSearchCV(best, grid, n_iter=10, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1)
search.fit(X_tr, y_tr)
tune_time = time.time() - t_tune
base_cv_auc = lb.iloc[0]["AUC"]
print("tuned CV AUC:", round(search.best_score_, 4), "vs base", round(base_cv_auc, 4))
final_pipe = search.best_estimator_ if search.best_score_ > base_cv_auc else best.fit(X_tr, y_tr)

# %%
def scores(model, X_, y_):
    p = model.predict(X_)
    s = model.predict_proba(X_)[:, 1]
    return {"Accuracy": accuracy_score(y_, p), "AUC": roc_auc_score(y_, s), "Recall": recall_score(y_, p),
            "Prec.": precision_score(y_, p), "F1": f1_score(y_, p),
            "Kappa": cohen_kappa_score(y_, p), "MCC": matthews_corrcoef(y_, p)}
hold = scores(final_pipe, X_te, y_te)
print("Hold-out:", {k: round(v, 4) for k, v in hold.items()})
ConfusionMatrixDisplay.from_estimator(final_pipe, X_te, y_te, cmap="Greens"); plt.savefig("trad_confusion.png", dpi=120); plt.close()

# %%
final_pipe.fit(X, y)  # finalize on train+test
joblib.dump(final_pipe, "trad_model.pkl")
un = scores(final_pipe, unseen.drop(columns="default"), unseen["default"])
total = time.time() - T0
print("Unseen accuracy:", round(un["Accuracy"], 4), "| total runtime (s):", round(total, 1))
lb.to_csv("trad_leaderboard.csv", index=False)
json.dump({"best": best_name, "tuned_cv_auc": search.best_score_, "base_cv_auc": base_cv_auc,
           "holdout": hold, "unseen": un, "compare_time": compare_time, "tune_time": tune_time,
           "total_time": total}, open("trad_summary.json", "w"), indent=2)
