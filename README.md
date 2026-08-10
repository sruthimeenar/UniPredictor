# JoSAA Compass

Run a local web server in this folder, then open `http://localhost:8000` in your browser:

```powershell
cd "C:\Users\purpl\Documents\Codex\2026-08-05\b\outputs"
python -m http.server 8000
```

The app reads `data/josaa_2024_round5.csv` and `data/college_company_mou.csv` at runtime. It uses supplied Round 5, the latest cut-off file in the archive; no Round 6 file was supplied. Career entries displayed when no matching MoU exists are deterministic, labelled synthetic examples.

`train_catboost.py` is the reproducible CatBoost pipeline for the Round 5 closing-rank estimator. Install its dependencies with `pip install -r requirements.txt` before running it; it writes a `.cbm` model and validation metrics alongside the script.
