# Credit Card Default Prediction with PyCaret

**Group 8:** Deven Bhole (S027), Syed Fazal Mehdi (S011), Manas Dabhade (S025) · 

Dataset: [UCI – Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (24,000-row version shipped with PyCaret, included in `data/credit.csv`).

## Run in VS Code

> PyCaret 3.3 needs **Python 3.9 – 3.11** (not 3.12+).

1. Open this folder in VS Code (`File → Open Folder`).
2. Open a terminal (`Ctrl + ``) and create a virtual environment:
   ```bash
   python -m venv .venv          # use py -3.11 -m venv .venv on Windows if you have several Pythons
   # Windows:   .venv\Scripts\activate
   # Mac/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. `Ctrl+Shift+P → Python: Select Interpreter` → pick `.venv`.
4. Train (≈3–6 min):
   ```bash
   python train.py
   ```
   Writes the leaderboard, metrics, plots and `credit_default_model.pkl` to `outputs/`.
5. Launch the demo:
   ```bash
   streamlit run app.py
   ```
   Opens at http://localhost:8501. Use the **Batch CSV** tab with `outputs/unseen_sample.csv`.

## Repository structure
```
├── train.py                 # PyCaret pipeline
├── app.py                   # Streamlit demo
├── data/credit.csv          # UCI dataset
├── outputs/                 # leaderboard, metrics, plots, saved model
├── traditional/             # scikit-learn notebook (Colab) for comparison
├── docs/                    # report (.docx) and presentation (.pptx)
└── requirements.txt
```

## Files
| File | Purpose |
|---|---|
| `train.py` | PyCaret pipeline: setup → compare_models → tune_model → evaluate → finalize → save |
| `app.py` | Streamlit web demo (single customer + batch CSV scoring) |
| `data/credit.csv` | Dataset |
| `docs/` | Project report and PPT |
| `traditional/` | Traditional scikit-learn notebook + results |
| `outputs/` | Leaderboard, metrics, AUC / confusion matrix / feature importance plots, saved model |

## Results (session_id = 42)
Best model: **Gradient Boosting Classifier**, 5-fold CV AUC 0.783, accuracy 0.821; hold-out AUC 0.759, accuracy 0.811; accuracy on 1,200 never-seen customers 0.827.

## Traditional baseline (for comparison)
`traditional/Traditional_sklearn_Colab.ipynb` rebuilds the same experiment by hand with scikit-learn + imbalanced-learn + LightGBM (upload to Google Colab, or run in VS Code). `trad.py` is the same code as a script.
Result: same best model (Gradient Boosting), CV AUC 0.773, hold-out accuracy 0.782, about 10.6 min runtime vs 5.8 min for PyCaret, and roughly 5x more modelling code.
