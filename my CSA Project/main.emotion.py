import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Mapping numbers to actual emotions based on your dataset
EMOTION_MAP = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

def run_project():
    # 1. Load Data (Note: using sep='\t' because your file is tab-separated)
    data_path = "data.csv" 
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # No header in your file, so we assign names manually
    df = pd.read_csv(data_path, sep='\t', header=None, names=["text", "label"])
    X = df["text"]
    y = df["label"]

    # 2. Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)

    # 3. Training
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"--- Emotion Detection Model Trained ---")
    print(f"Dataset size: {len(df)} rows")
    print(f"Validation Accuracy: {acc:.2%}")

    # 5. Handling Command Line Input
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        vec = vectorizer.transform([input_text])
        pred_num = model.predict(vec)[0]
        emotion = EMOTION_MAP.get(pred_num, "Unknown")
        print(f"\nText: '{input_text}'")
        print(f"Predicted Emotion: {emotion}")
    else:
        print("\n[Interactive Mode] Type 'exit' to stop.")
        while True:
            text = input("\nEnter text: ")
            if text.lower() == "exit": break
            vec = vectorizer.transform([text])
            pred_num = model.predict(vec)[0]
            emotion = EMOTION_MAP.get(pred_num, "Unknown")
            print(f"Predicted Emotion: {emotion}")

if __name__ == "__main__":
    run_project()