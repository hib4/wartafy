from preprocess import clean_text
from train import load_models


def predict_single(text: str, model_name: str = "Logistic Regression") -> dict:
    models = load_models()

    if not models:
        raise RuntimeError("No trained models found. Run train.py first.")

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(models.keys())}")

    pipeline = models[model_name]
    cleaned = clean_text(text)

    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_

    confidence_map = dict(zip(classes, proba))
    confidence = confidence_map[pred]

    return {
        "category": pred,
        "confidence": round(confidence * 100, 2),
        "all_probabilities": {k: round(v * 100, 2) for k, v in confidence_map.items()},
    }


def predict_both_models(text: str) -> dict:
    models = load_models()
    results = {}
    for name in models:
        results[name] = predict_single(text, model_name=name)
    return results


if __name__ == "__main__":
    samples = [
        "NASA launches new telescope to observe distant galaxies and black holes.",
        "The Lakers beat the Celtics 112-98 in an intense NBA playoff game last night.",
        "Parliament passes new budget with increased spending on healthcare and education.",
        "Apple unveils its latest iPhone model with improved AI camera features.",
    ]

    models = load_models()
    if not models:
        print("No models found. Please run train.py first.")
    else:
        print("Demo Predictions\n" + "=" * 50)
        for text in samples:
            print(f"\nText     : {text[:70]}...")
            result = predict_single(text)
            print(f"Category : {result['category']}  (Confidence: {result['confidence']}%)")
            print(f"All probs: { {k: f'{v}%' for k, v in result['all_probabilities'].items()} }")