# SMS Spam Classification

This project is a simple SMS spam classifier using machine learning. It trains a **Naive Bayes** model to classify SMS messages as either **ham** (normal message) or **spam**. The project also provides a **prediction script** and a **Flask API** to test messages interactively.

---

## 📂 Project Structure
```
sms-spam-classification/
├─ data/
│ ├─ spam_cleaned.csv # Preprocessed dataset
| ├─ spam.csv #the real dataset
├─ models/ # Saved models and vectorizer
│ ├─ naive_bayes.pkl
│ ├─ logistic_regression.pkl
│ └─ vectorizer.pkl
├─ src/
| ├─ exploratory_analysis.py #to show number of samples per category, most frequent words
| ├─ load_and_eda.py #This was created just to test the dataset
| ├─ preprocess.py # Basic preprocessing done here
│ ├─ model_training.py # Script to train models
│ ├─ predict.py # Simple terminal prediction script
│ └─ app.py # Flask API to test predictions
└─ README.md
```

---

## 📝 Dataset

- The dataset (`spam_cleaned.csv`) contains SMS messages and labels (`ham` or `spam`).  
- Messages were **cleaned, lowercased, and preprocessed** before training.
- The dataset link : https://www.kaggle.com/datasets/team-ai/spam-text-message-classification

---

## 🔍 Exploratory Analysis

- Displayed **number of messages per category** and **the most frequent words in the messages** :

```python
df['Label'].value_counts()
```
- Checked the 10 most frequent words in the dataset using collections.Counter.
- Optional: visualized category counts using a bar chart. Screenshot included.

---

## ⚙️ Model Training

- **Models trained:** Naive Bayes (NB) and Logistic Regression (LR)

- **TF-IDF** was used to convert text messages into numerical features.

- **Dataset was split:** 80% training, 20% testing.

- **Evaluation metrics:** accuracy, precision, recall, F1-score.

- Confusion matrices were visualized using Seaborn.

---

## 💻 Prediction Script (predict.py)
- Allows users to enter a message in the terminal and get predictions (ham or spam).

**Usage:** ```bash python predict.py ```

**Example:**
Enter a message(or type 'quit' to exit): 07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow,

Prediction: spam

---

## 🌐 Flask API (app.py)
- Provides a web API endpoint to test predictions without running Python scripts directly.

- The endpoint /predict accepts POST requests with JSON:
{
    "message": "Your text message here"
}
- Returns prediction as JSON:
{
  "prediction": "ham"  // or "spam"
}

**Usage:**

1. Start the Flask API:
```bash
python app.py
```

2. Test the endpoint using curl:
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d "{\"message\": \"Congrats! You won a free iPhone.\"}"
```
---

## 👩🏻‍💻 Author
Xhesika Mula
