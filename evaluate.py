import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_loader import load_ag_news
from train import load_models, train_and_save

OUTPUT_DIR = "outputs"


def evaluate_model(name: str, pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    return {
        "name": name,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "classes": list(pipeline.classes_),
        "y_pred": y_pred,
    }


def print_report(result: dict):
    print(f"\n{'=' * 55}")
    print(f"  {result['name']}")
    print(f"{'=' * 55}")
    print(f"  Accuracy : {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print()

    report = result["report"]
    classes = result["classes"]

    print(f"  {'Category':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-'*62}")
    for cls in classes:
        r = report[cls]
        print(f"  {cls:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10}")

    print(f"  {'-'*62}")
    ma = report["macro avg"]
    print(f"  {'Macro Average':<20} {ma['precision']:>10.4f} {ma['recall']:>10.4f} {ma['f1-score']:>10.4f}")
    wa = report["weighted avg"]
    print(f"  {'Weighted Average':<20} {wa['precision']:>10.4f} {wa['recall']:>10.4f} {wa['f1-score']:>10.4f}")


def plot_confusion_matrix(result: dict, save_path: str):
    cm = result["confusion_matrix"]
    classes = result["classes"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {result['name']}", fontsize=14, fontweight="bold", y=1.02)

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Counts", "Percentages (%)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


def plot_model_comparison(results: list[dict], save_path: str):
    model_names = [r["name"] for r in results]
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    data = []
    for r in results:
        ma = r["report"]["macro avg"]
        data.append([
            r["accuracy"],
            ma["precision"],
            ma["recall"],
            ma["f1-score"],
        ])

    x = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, color, vals) in enumerate(zip(labels, colors, zip(*data))):
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_title("Model Comparison — Macro-Averaged Metrics", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.axhline(y=0.75, color="red", linestyle="--", linewidth=1, label="Success threshold (0.75)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Model comparison chart saved -> {save_path}")


def run_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading dataset...")
    train_df, test_df = load_ag_news(max_train=20_000, max_test=2_000)
    models = load_models()
    if not models:
        print("No saved models found — training now...")
        models = train_and_save(train_df)

    X_test = test_df["clean_text"]
    y_test = test_df["category"]

    results = []
    for name, pipeline in models.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_model(name, pipeline, X_test, y_test)
        results.append(result)
        print_report(result)

        safe_name = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            result,
            os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe_name}.png")
        )

    if len(results) > 1:
        print("\nGenerating model comparison chart...")
        plot_model_comparison(results, os.path.join(OUTPUT_DIR, "model_comparison.png"))
    print(f"\n{'=' * 45}")
    print("  SUMMARY")
    print(f"{'=' * 45}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*43}")
    for r in results:
        f1 = r["report"]["macro avg"]["f1-score"]
        status = "PASS" if r["accuracy"] >= 0.75 else "FAIL"
        print(f"  {r['name']:<25} {r['accuracy']:>10.4f} {f1:>10.4f}  [{status}]")

    return results


if __name__ == "__main__":
    run_evaluation()