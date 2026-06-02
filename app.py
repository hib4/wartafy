import os
import sys
import glob
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

st.set_page_config(
    page_title="Wartafy",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title  { font-size: 2.2rem; font-weight: 700; color: #1F4E79; }
    .sub-title   { font-size: 1rem; color: #595959; margin-bottom: 1.5rem; }
    .result-box  { padding: 1.2rem 1.5rem; border-radius: 10px; margin: 0.5rem 0; }
    .badge       { display: inline-block; padding: 0.35rem 1rem; border-radius: 20px;
                   font-weight: 700; font-size: 1.1rem; }
    .section-sep { border-top: 2px solid #E0E0E0; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

CATEGORY_COLORS = {
    "Sports":           "#2196F3",
    "Science/Technology": "#4CAF50",
    "Business":         "#FF9800",
    "World":            "#9C27B0",
}
CATEGORY_ICONS = {
    "Sports":             "⚽",
    "Science/Technology": "🔬",
    "Business":           "💼",
    "World":              "🌍",
}


@st.cache_resource(show_spinner="Loading models...")
def get_models():
    from train import load_models, LabelDecodingPipeline
    return load_models()


def run_prediction(text: str):
    from predict import predict_both_models
    return predict_both_models(text)


def plot_confidence_bars(result: dict, model_name: str):
    probs = result["all_probabilities"]
    cats  = list(probs.keys())
    vals  = list(probs.values())
    colors = [CATEGORY_COLORS.get(c, "#90CAF9") for c in cats]

    fig, ax = plt.subplots(figsize=(5, 2.8))
    bars = ax.barh(cats, vals, color=colors, height=0.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.set_title(model_name, fontsize=10, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.axvline(x=result["confidence"], color="red", linestyle="--",
               linewidth=1, alpha=0.5, label=f"Top: {result['confidence']}%")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=72)
    st.title("Settings")
    selected_model = st.selectbox(
        "Primary model",
        ["Logistic Regression", "Naive Bayes", "Neural Network", "XGBoost", "All (compare)"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "This classifier uses TF-IDF feature extraction with Logistic Regression, "
        "Multinomial Naive Bayes, Neural Network (MLP), and XGBoost, trained on the AG News dataset."
    )
    st.markdown("**Categories**")
    for cat, icon in CATEGORY_ICONS.items():
        st.caption(f"{icon} {cat}")

    st.markdown("---")
    if st.button("Train / Retrain Models", type="secondary", use_container_width=True):
        with st.spinner("Training models (this may take a minute)..."):
            from data_loader import load_ag_news
            from train import train_and_save
            train_df, _ = load_ag_news(max_train=20_000, max_test=2_000)
            train_and_save(train_df)
            st.cache_resource.clear()
        st.success("Models trained and saved!")

    if st.button("Run Evaluation", type="secondary", use_container_width=True):
        with st.spinner("Evaluating models..."):
            from evaluate import run_evaluation
            run_evaluation()
        st.success("Evaluation complete! Check the Evaluation tab.")


tab_predict, tab_batch, tab_eval, tab_about = st.tabs([
    "🔍 Predict", "📋 Batch Predict", "📊 Evaluation", "ℹ️ About"
])


with tab_predict:
    st.markdown('<p class="main-title">📰 Wartafy</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Paste or type a news article and the model will classify its category.</p>', unsafe_allow_html=True)

    sample_articles = {
        "Select a sample...": "",
        "Sports": "The Golden State Warriors defeated the Boston Celtics 118-102 in Game 6, clinching the NBA Championship title. Stephen Curry was named Finals MVP after averaging 31 points per game throughout the series.",
        "Science/Technology": "Scientists at MIT have developed a new battery technology that can charge electric vehicles in under 5 minutes while maintaining over 80% capacity after 1,000 charge cycles, potentially revolutionizing the EV industry.",
        "Business": "Amazon reported record quarterly earnings of $143 billion in revenue, driven by strong growth in its AWS cloud division and advertising business. The company also announced plans to hire 100,000 new workers globally.",
        "World": "World leaders gathered in Geneva for emergency climate talks following a series of catastrophic floods across Southeast Asia. Representatives from 40 nations pledged to accelerate carbon reduction timelines.",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        sample_choice = st.selectbox("Load a sample article", list(sample_articles.keys()))
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

    default_text = sample_articles.get(sample_choice, "")
    input_text = st.text_area(
        "Enter news article text",
        value=default_text,
        height=160,
        placeholder="Paste or type a news article here...",
    )

    predict_btn = st.button("Classify Article", type="primary", use_container_width=False)

    if predict_btn:
        if not input_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            models = get_models()
            if not models:
                st.error("No trained models found. Use the sidebar to train models first.")
            else:
                with st.spinner("Classifying..."):
                    results = run_prediction(input_text)

                models_to_show = (
                    list(results.keys()) if selected_model == "All (compare)"
                    else [selected_model] if selected_model in results
                    else list(results.keys())
                )

                st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

                if len(models_to_show) == 1:
                    name = models_to_show[0]
                    r = results[name]
                    cat = r["category"]
                    color = CATEGORY_COLORS.get(cat, "#90CAF9")
                    icon  = CATEGORY_ICONS.get(cat, "📄")

                    st.markdown(f"""
                    <div class="result-box" style="background:{color}18; border-left: 5px solid {color};">
                        <span class="badge" style="background:{color}; color:white;">{icon} {cat}</span>
                        &nbsp;&nbsp;<span style="color:#595959;">Confidence: <b>{r['confidence']}%</b> &nbsp;|&nbsp; Model: <b>{name}</b></span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.pyplot(plot_confidence_bars(r, name))

                else:
                    for name in models_to_show:
                        r = results[name]
                        cat = r["category"]
                        color = CATEGORY_COLORS.get(cat, "#90CAF9")
                        icon  = CATEGORY_ICONS.get(cat, "📄")
                        st.markdown(f"""
                        <div class="result-box" style="background:{color}18; border-left: 5px solid {color};">
                            <span class="badge" style="background:{color}; color:white;">{icon} {cat}</span>
                            &nbsp;&nbsp;<span style="color:#595959;">Confidence: <b>{r['confidence']}%</b> &nbsp;|&nbsp; Model: <b>{name}</b></span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.pyplot(plot_confidence_bars(r, name))


with tab_batch:
    st.header("Batch Prediction")
    st.caption("Upload a CSV file with a column named `text` to classify multiple articles at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must have a column named 'text'.")
        else:
            models = get_models()
            if not models:
                st.error("No trained models found. Please train models first.")
            else:
                model_choice = st.selectbox("Model for batch prediction",
                                            list(models.keys()), key="batch_model")
                if st.button("Run Batch Classification", type="primary"):
                    from predict import predict_single
                    from preprocess import clean_text

                    progress = st.progress(0)
                    preds = []
                    for i, row in enumerate(df["text"]):
                        r = predict_single(str(row), model_name=model_choice)
                        preds.append({"category": r["category"], "confidence (%)": r["confidence"]})
                        progress.progress((i + 1) / len(df))

                    result_df = pd.concat([df, pd.DataFrame(preds)], axis=1)
                    st.success(f"Classified {len(result_df)} articles!")
                    st.dataframe(result_df, use_container_width=True)

                    csv = result_df.to_csv(index=False).encode()
                    st.download_button("Download Results CSV", csv,
                                       "classified_news.csv", "text/csv")

                    fig, ax = plt.subplots(figsize=(5, 3))
                    counts = result_df["category"].value_counts()
                    colors = [CATEGORY_COLORS.get(c, "#90CAF9") for c in counts.index]
                    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
                           colors=colors, startangle=140)
                    ax.set_title("Category Distribution", fontweight="bold")
                    st.pyplot(fig)


with tab_eval:
    st.header("Model Evaluation Results")

    OUTPUT_DIR = "outputs"
    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    lr_cm_path      = os.path.join(OUTPUT_DIR, "confusion_matrix_logistic_regression.png")
    nb_cm_path      = os.path.join(OUTPUT_DIR, "confusion_matrix_naive_bayes.png")

    if not os.path.exists(comparison_path):
        st.info("No evaluation results found yet. Click 'Run Evaluation' in the sidebar.")
    else:
        st.subheader("Model Comparison")
        st.image(comparison_path, use_column_width=True)

        st.subheader("Confusion Matrices")
        cm_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "confusion_matrix_*.png")))
        cols = st.columns(2)
        for i, path in enumerate(cm_files):
            col = cols[i % 2]
            model_name = os.path.basename(path).replace("confusion_matrix_", "").replace(".png", "").replace("_", " ").title()
            with col:
                st.image(path, caption=model_name, use_column_width=True)


with tab_about:
    st.header("About This Project")
    st.markdown("""
    ### Wartafy
    An NLP-based automatic text classification system built as part of a machine learning coursework project.

    #### How It Works
    1. **Text Preprocessing** — Lowercasing, URL/HTML removal, punctuation stripping, stopword removal via NLTK
    2. **Feature Extraction** — TF-IDF vectorization (unigrams + bigrams, up to 50,000 features)
    3. **Classification** — Logistic Regression, Multinomial Naive Bayes, Neural Network (MLP), and XGBoost
    4. **Evaluation** — Accuracy, Precision, Recall, F1-Score, and Confusion Matrix

    #### Dataset
    **AG News Classification Dataset** — 120,000 training / 7,600 test articles across 4 categories:
    World, Sports, Business, and Science/Technology.
    Source: [HuggingFace Datasets](https://huggingface.co/datasets/fancyzhx/ag_news)

    #### Technology Stack
    | Component | Library |
    |-----------|---------|
    | Language | Python 3.x |
    | NLP | NLTK |
    | ML | Scikit-learn, XGBoost |
    | Data | Pandas, HuggingFace Datasets |
    | UI | Streamlit |
    | Plots | Matplotlib, Seaborn |

    #### Success Criteria
    - Model accuracy >= 75%
    - Macro F1-score >= 0.75
    - Prediction time < 2 seconds
    """)
