import pandas as pd
import sys
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_project():
    # 1. Load dataset
    data_path = "data.csv"

    if not os.path.exists(data_path):
        print("❌ Error: data.csv not found in this folder")
        return

    # Read dataset (tab-separated)
    df = pd.read_csv(data_path, sep="\t", header=None)
    df.columns = ["text", "label"]

    # 2. Clean text + NEGATION FIX
    def clean_text(text):
        text = text.lower()

        # 🔥 Handle negation
        text = text.replace("not ", "not_")

        # Remove special characters
        text = re.sub(r'[^a-z_\s]', '', text)

        return text

    df["text"] = df["text"].apply(clean_text)

    # 3. Map 6 emotions → 4 emotions
    def map_emotion(label):
        if label == 0:
            return "Sadness"
        elif label == 1 or label == 2:
            return "Joy"
        elif label == 3:
            return "Anger"
        elif label == 4:
            return "Sadness"
        else:
            return "Neutral"

    df["emotion"] = df["label"].apply(map_emotion)

    # 4. Features and labels
    X = df["text"]
    y = df["emotion"]

    # 5. Vectorization (no stopwords removed)
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X_vectorized = vectorizer.fit_transform(X)

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # 7. Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 8. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n✅ Model Trained Successfully")
    print(f"Dataset size: {len(df)} rows")
    print(f"Accuracy: {accuracy:.2%}")

    # 9. Interactive mode
    print("\nType 'exit' to stop")
    while True:
        text = input("\nEnter text: ")
        if text.lower() == "exit":
            break

        text_clean = clean_text(text)
        vec = vectorizer.transform([text_clean])
        emotion = model.predict(vec)[0]

        print(f"Predicted Emotion: {emotion}")

# Run program
if __name__ == "__main__":
    run_project()
