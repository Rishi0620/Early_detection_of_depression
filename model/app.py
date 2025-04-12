import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from pathlib import Path

nltk.download('stopwords')

MODEL_DIR = Path('model_results')
MODEL_PATHS = {
    'svm': MODEL_DIR / 'svm_model.pkl',
    'random_forest': MODEL_DIR / 'random_forest_model.pkl',
    'logistic_regression': MODEL_DIR / 'logistic_regression_model.pkl'
}

CLASS_DESCRIPTIONS = {
    0: "Neutral/Positive Content",
    1: "Potential Stress/Anxiety Indicators",
    2: "Potential Depression Indicators",
    3: "Potential Loneliness/Social Isolation",
    4: "General Mental Health Discussion"
}

@st.cache_resource
def load_components(model_choice):
    try:
        model = joblib.load(MODEL_PATHS[model_choice])

        tfidf = joblib.load('models/tfidf.pkl')
        svd = joblib.load('models/svd.pkl')

        return model, tfidf, svd
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()


def preprocess_text(text):
    if isinstance(text, float):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)


def extract_features(texts, tfidf, svd):
    cleaned_texts = [preprocess_text(text) for text in texts]
    tfidf_features = tfidf.transform(cleaned_texts)
    svd_features = svd.transform(tfidf_features)
    return svd_features


st.set_page_config(
    page_title="Mental Health Text Analyzer",
    page_icon="",
    layout="wide"
)

with st.sidebar:
    st.title("Settings")
    model_choice = st.radio(
        "Select Model:",
        options=['svm', 'random_forest', 'logistic_regression'],
        format_func=lambda x: {
            'svm': 'SVM (Most Accurate)',
            'random_forest': 'Random Forest (Balanced)',
            'logistic_regression': 'Logistic Regression (Fastest)'
        }[x],
        key='model_choice'
    )

    st.markdown("---")
    st.markdown("**Model Performance:**")
    performance_data = {
        'Model': ['SVM', 'Random Forest', 'Logistic Regression'],
        'Accuracy': [0.724, 0.704, 0.669],
        'F1 Score': [0.726, 0.704, 0.668],
        'Speed': ['Slow', 'Medium', 'Fast']
    }
    st.dataframe(pd.DataFrame(performance_data), hide_index=True)

    st.markdown("---")
    st.markdown("""
    **About this app:**
    - Analyzes text for mental health indicators
    - Uses models trained on mental health discussions
    - Categories: 0=Neutral, 1=Anxiety, 2=Depression, 3=Loneliness, 4=General
    """)

st.title("Mental Health Text Analyzer")
st.markdown("Analyze text for mental health indicators using different ML models")

if 'current_model' not in st.session_state or st.session_state.current_model != model_choice:
    model, tfidf, svd = load_components(model_choice)
    st.session_state.current_model = model_choice
    st.session_state.model = model
    st.session_state.tfidf = tfidf
    st.session_state.svd = svd

user_input = st.text_area(
    "Enter text to analyze:",
    height=200,
    placeholder="Type or paste text here...",
    key="user_input"
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Analyze Text", type="primary"):
        if user_input.strip():
            with st.spinner(f"Analyzing with {model_choice.replace('_', ' ').title()}..."):
                try:
                    features = extract_features(
                        [user_input],
                        st.session_state.tfidf,
                        st.session_state.svd
                    )

                    prediction = st.session_state.model.predict(features)[0]
                    probabilities = st.session_state.model.predict_proba(features)[0]

                    st.success("Analysis Complete")
                    st.markdown("---")

                    st.subheader("Primary Prediction")
                    confidence = probabilities[prediction]
                    st.metric(
                        label="Predicted Category",
                        value=CLASS_DESCRIPTIONS[prediction],
                        help=f"Model confidence: {confidence:.1%}"
                    )

                    with st.expander("View Detailed Probabilities"):
                        prob_df = pd.DataFrame({
                            "Category": [CLASS_DESCRIPTIONS[i] for i in range(5)],
                            "Probability": probabilities
                        }).sort_values("Probability", ascending=False)

                        st.dataframe(prob_df, hide_index=True)

                        for i, prob in enumerate(probabilities):
                            st.progress(prob, text=f"{CLASS_DESCRIPTIONS[i]}: {prob:.1%}")

                    st.markdown("---")
                    st.subheader("Interpretation Guide")
                    st.markdown("""
                    - **0 (Neutral/Positive)**: Generally positive or neutral content
                    - **1 (Anxiety)**: Signs of stress, worry, or anxiety
                    - **2 (Depression)**: Indicators of sadness or depression
                    - **3 (Loneliness)**: Signs of isolation or lack of social connection
                    - **4 (General)**: Mental health discussion without specific indicators
                    """)

                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.warning("Please enter some text to analyze")

with col2:
    st.markdown("**Example Inputs:**")
    st.code("""
    # Class 0 (Neutral/Positive)
    "I've been feeling great lately and enjoying time with friends"

    # Class 1 (Anxiety)
    "I can't stop worrying about everything, my heart races constantly"

    # Class 2 (Depression)
    "Nothing brings me joy anymore, I feel empty inside"

    # Class 3 (Loneliness)
    "I spend all my time alone, no one reaches out to me"

    # Class 4 (General)
    "Mental health awareness is important for everyone"
    """)

    st.markdown("**Current Model Info:**")
    if model_choice == 'svm':
        st.info("""
        **SVM Model (Best Performance)**
        - Accuracy: 72.4%
        - Best for: Most accurate predictions
        - Parameters: C=1, kernel=rbf, gamma=auto
        """)
    elif model_choice == 'random_forest':
        st.info("""
        **Random Forest Model**
        - Accuracy: 70.4%
        - Best for: Balanced speed/accuracy
        - Parameters: 200 trees, max_depth=20
        """)
    else:
        st.info("""
        **Logistic Regression Model**
        - Accuracy: 66.9%
        - Best for: Fastest predictions
        - Parameters: C=0.01, penalty=l2
        """)

st.markdown("---")
st.caption("""
Note: This tool is not a substitute for professional mental health advice. 
The models were trained on limited data and may not capture all problems.
""")