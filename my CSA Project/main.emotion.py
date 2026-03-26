import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Mapping labels (your dataset uses numbers 0–5)
EMOTION_MAP = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

def run_project():
    # 1. Load dataset
    data_path = "data.csv"

    if not os.path.exists(data_path):
        print("❌ Error: data.csv not found in this folder")
        return

    # Read tab-separated file
    df = pd.read_csv(data_path, sep="\t", header=None)
    df.columns = ["text", "label"]

    # 2. Split features and labels
    X = df["text"]
    y = df["label"]

    # 3. Convert text to numbers
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # 5. Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n✅ Model Trained Successfully")
    print(f"Dataset size: {len(df)} rows")
    print(f"Accuracy: {accuracy:.2%}")

    # 7. Interactive prediction
    print("\nType 'exit' to stop")
    while True:
        text = input("\nEnter text: ")
        if text.lower() == "exit":
            break

        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        emotion = EMOTION_MAP.get(pred, "Unknown")

        print(f"Predicted Emotion: {emotion}")

# Run program
if __name__ == "__main__":
    run_project()
