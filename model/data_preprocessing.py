import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


def preprocess_text(text):
    # Check if text is NaN (float type) or not a string
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text


def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # Ensure the 'text' column exists and fill NaN values with empty strings
    if 'text' not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")

    df['text'] = df['text'].fillna('').astype(str)
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Saved to {output_file}")


if __name__ == "__main__":
    input_file = '/Users/rishabhbhargav/PycharmProjects/Rishi0620/NLP/data/raw/data_to_be_cleansed.csv'
    output_file = '/Users/rishabhbhargav/PycharmProjects/Rishi0620/NLP/data/processed/clean_reddit_data.csv'
    preprocess_data(input_file, output_file)