import pandas as pd
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
df.columns = ["label", "text"]

# Encode labels (ham = 0, spam = 1)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Text preprocessing
def preprocess(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    return text

df["text"] = df["text"].apply(preprocess)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Build pipeline with TF-IDF
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "spam_tfidf_model.pkl")

# Custom prediction function
def predict_spam(text):
    clean_text = preprocess(text)
    pred = model.predict([clean_text])
    return "Spam" if pred[0] == 1 else "Not Spam"

# Example usage
print("Custom prediction:", predict_spam("Hey! You've won a free prize. Claim now!"))
