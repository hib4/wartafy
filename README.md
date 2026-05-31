# Wartafy

Compact end-to-end run guide for new users.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- NLTK resources are downloaded automatically on first run.
- Data lives in `data/train.csv` and `data/test.csv` (AG News).

## 2) Train models

```bash
python train.py
```

Trained models are saved to `models/` as `*.pkl`.

## Git LFS for model files

This project uses Git LFS for trained model files in `models/*.pkl`.

Before cloning or pulling model files, contributors should install and initialize Git LFS:

```bash
git lfs install
```

If Git LFS is not installed yet, install it first with your package manager, then run the command above.

After cloning, or after pulling commits that update model files, download the LFS-managed files with:

```bash
git lfs pull
```

## 3) Evaluate models

```bash
python evaluate.py
```

Evaluation outputs (confusion matrices and a comparison chart) are saved to `outputs/`.

## 4) Run the app (UI)

```bash
streamlit run app.py
```

In the sidebar, you can:

- Train/retrain models
- Run evaluation
- Pick which model to use for prediction

## 5) Run predictions from the CLI

Single prediction (default model):

```bash
python predict.py
```

Programmatic use example:

```python
from predict import predict_single

result = predict_single("NASA launches a new satellite")
print(result)
```

## Troubleshooting

- If you see "No trained models found", run `python train.py` first.
- If NLTK downloads fail, re-run any script; downloads are retried automatically.
