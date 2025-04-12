import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib

def extract_features(input_file, output_file):
    df = pd.read_csv(input_file)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])

    svd = TruncatedSVD(n_components=300)
    X_svd = svd.fit_transform(X)

    joblib.dump(vectorizer, 'models/tfidf.pkl')
    joblib.dump(svd, 'models/svd.pkl')

    joblib.dump(X_svd, output_file)
    print(f"Features extracted and saved to {output_file}")


if __name__ == "__main__":
    input_file = 'data/processed/clean_reddit_data.csv'
    output_file = 'data/processed/extracted_features.pkl'
    extract_features(input_file, output_file)