# 📧 Spam Email Detector

A Machine Learning project that classifies emails as **Spam** or **Not Spam (Ham)** using **Naive Bayes** and **Natural Language Processing (NLP)**.

---

## 🚀 Project Description

Spam emails often contain fake offers, scams, or phishing messages.  
This project builds a **Spam Email Detector** that automatically identifies whether an email is spam or not using machine learning techniques.

The model is trained using the **Naive Bayes algorithm** and text preprocessing methods such as tokenization and stopword removal.

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Naive Bayes Classifier
- Natural Language Processing (NLP)
- Pandas, NumPy
- Joblib

---

## 📂 Project Structure

Spam_Email_Detector/
│
├── dataset/
│ └── spam.csv
│
├── model/
│ └── spam_model.pkl
│
├── train_model.py
├── spam_detector.py
├── requirements.txt
└── README.md

---

## 📊 Dataset

- **Name:** SMS Spam Collection Dataset  
- **Labels:**
  - `spam` → Unwanted or fraudulent email
  - `ham` → Legitimate email

---

## ⚙️ How the Project Works

1. Email text is cleaned (lowercase, punctuation removal, stopword removal).
2. Text is converted into numerical features using **CountVectorizer**.
3. A **Naive Bayes model** is trained on the dataset.
4. The trained model is saved as `spam_model.pkl`.
5. New email messages are classified using the saved model.

---

## ▶️ How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt

---
### Step 2: Train the model
python train_model.py

---

### Step 3: Test the spam detector
python spam_detector.py

---

### Example Output
Enter email message: Free money offer just for you!
🚫 Spam Email

### Enter email message: Are we meeting tomorrow?
✅ Not Spam Email
