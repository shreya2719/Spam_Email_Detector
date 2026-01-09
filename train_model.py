import pandas as pd
import nltk
import joblib
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

nltk.download('stopwords')

data = pd.read_csv('dataset/spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

data['message'] = data['message'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

model.fit(X_train, y_train)

joblib.dump(model, 'model/spam_model.pkl')

print("✅ Model trained and saved")
