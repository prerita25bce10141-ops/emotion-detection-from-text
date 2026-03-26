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

    df = pd.read_csv(data_path, sep="\t", header=None)
    df.columns = ["text", "label"]

    # 2. Clean text + negation handling
    def clean_text(text):
        text = text.lower()
        text = text.replace("not ", "not_")   # handle negation
        text = re.sub(r'[^a-z_\s]', '', text)
        return text

    df["text"] = df["text"].apply(clean_text)

    # 3. Map labels to emotions
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

    # 4. Features
    X = df["text"]
    y = df["emotion"]

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X_vectorized = vectorizer.fit_transform(X)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # 6. Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n✅ Model Trained Successfully")
    print(f"Dataset size: {len(df)} rows")
    print(f"Accuracy: {accuracy:.2%}")

    # 🔥 Rule-based fixes
    def rule_based_fix(text):
        text = text.lower()

        # Negation cases
        if "not confident" in text or "not happy" in text or "not satisfied" in text:
            return "Sadness"

        # Fear / anxiety
        if "scared" in text or "afraid" in text or "nervous" in text:
            return "Sadness"

        # Anger cases
        if "angry" in text or "mad" in text or "furious" in text:
            return "Anger"

        return None

    # 8. Interactive mode
    print("\nType 'exit' to stop")
    while True:
        text = input("\nEnter text: ")
        if text.lower() == "exit":
            break

        # Apply rule-based fix first
        rule = rule_based_fix(text)

        if rule:
            emotion = rule
        else:
            text_clean = clean_text(text)
            vec = vectorizer.transform([text_clean])
            emotion = model.predict(vec)[0]

        print(f"Predicted Emotion: {emotion}")


# Run program
if __name__ == "__main__":
    run_project()
