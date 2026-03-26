# Emotion Detection from Text (CLI-Based ML Project)

## 📌 Project Overview

This project is a command-line based Emotion Detection system built using Machine Learning. It analyzes user-input text and predicts the underlying emotion.

The model is trained on a labeled dataset of text samples and uses Natural Language Processing (NLP) techniques to classify emotions.

---

## 🎯 Problem Statement

Understanding emotions in text is important for applications like:

* Mental health analysis
* Feedback systems
* Chatbots and virtual assistants

However, basic machine learning models struggle with context and negation (e.g., “not happy”). This project explores these challenges and implements improvements to handle them.

---

## 🧠 Approach

1. **Dataset**

   * Text dataset with labeled emotions (0–5)
   * Preprocessed into 4 main emotions:

     * Sadness
     * Joy
     * Anger
     * Neutral

2. **Preprocessing**

   * Lowercasing text
   * Removing special characters
   * Handling negation (e.g., "not happy" → "not_happy")

3. **Feature Extraction**

   * TF-IDF Vectorization
   * Includes unigrams and bigrams

4. **Model**

   * Logistic Regression classifier

5. **Enhancement**

   * Hybrid approach (ML + rule-based fixes)
   * Improves predictions for:

     * Negation phrases
     * Anxiety-related words (e.g., “scared”, “nervous”)

---

## 🛠️ Technologies Used

* Python
* pandas
* scikit-learn
* Regular Expressions (re)

---

## 📂 Project Structure

```
emotion-detection-from-text/
│── data.csv
│── main.emotion.py
│── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Python

Download and install Python (3.x) from the official website.

Make sure to check:

```
Add Python to PATH
```

---

### 2. Install Required Libraries

Open terminal / command prompt and run:

```
pip install pandas scikit-learn
```

---

### 3. Clone or Download Repository

```
git clone <your-repo-link>
cd emotion-detection-from-text
```

OR download ZIP and extract.

---

### 4. Ensure Dataset Placement

Make sure `data.csv` is in the same folder as:

```
main.emotion.py
```

---

## ▶️ How to Run

Open terminal inside project folder and run:

```
python main.emotion.py
```

---

## 💬 Usage

After running, the program will prompt:

```
Enter text:
```

Type any sentence:

Example:

```
i am feeling very happy today
```

Output:

```
Predicted Emotion: Joy
```

Type:

```
exit
```

to stop the program.

---

## 📊 Sample Output

```
Model Trained Successfully
Dataset size: 699 rows
Accuracy: ~60%

Enter text: i am scared
Predicted Emotion: Sadness
```

---

## ⚠️ Limitations

* The model uses TF-IDF (bag-of-words), so it:

  * struggles with context
  * may misinterpret complex sentences
* Negation handling is partially improved using preprocessing and rules
* Accuracy depends heavily on dataset quality

---

## 🚀 Future Improvements

* Use advanced NLP models (LSTM, BERT)
* Improve dataset size and diversity
* Build a web interface (Streamlit)
* Add visualization (confusion matrix)

---

## 📚 Learning Outcomes

Through this project, the following concepts were applied:

* Text preprocessing
* Feature extraction (TF-IDF)
* Classification using Logistic Regression
* Model evaluation
* Handling real-world ML limitations

---

## 👤 Author

Prerita Kukreja
First Year CSE Student

---

## ✅ Conclusion

This project demonstrates a complete ML pipeline for emotion detection and highlights the challenges of working with natural language. It combines machine learning with practical fixes to improve performance in real-world scenarios.
