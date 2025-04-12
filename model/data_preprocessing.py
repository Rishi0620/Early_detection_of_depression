import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = text.split()

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    cleaned_text = ' '.join(tokens)

    return cleaned_text


def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)

    df['cleaned_text'] = df['text'].apply(preprocess_text)

    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Saved to {output_file}")


if __name__ == "__main__":
    input_file = 'data/raw/data_to_be_cleansed.csv'
    output_file = 'data/processed/clean_reddit_data.csv'
    preprocess_data(input_file, output_file)