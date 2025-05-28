import pandas as pd
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
df.columns = ["label", "text"]

# Label encoding
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["text"] = df["text"].apply(preprocess)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Create pipeline
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "spam_classifier.pkl")

# Predict on custom input
def predict_spam(text):
    clean_text = preprocess(text)
    pred = model.predict([clean_text])
    return "Spam" if pred[0] == 1 else "Not Spam"

# Example
print("Custom prediction: ", predict_spam("Congratulations! You've won a free ticket to Bahamas. Call now!"))
