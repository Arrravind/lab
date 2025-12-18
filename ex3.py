# pip install seaborn scikit-learn


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

df = pd.read_csv("./datasets/IMDB-Dataset.csv")
df.dropna(inplace=True)

print("\nDataset Loaded Successfully")
print("Total Reviews:", len(df))
print("\nFirst 5 Records:")
print(df.head())

df["sentiment"] = df["sentiment"].str.capitalize()

def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

df["cleaned_review"] = df["review"].apply(clean_text)

print("\nSample Cleaned Review:\n", df["cleaned_review"].iloc[0])

plt.figure(figsize=(6, 4))
sns.countplot(x="sentiment", data=df, palette="coolwarm")
plt.title("Distribution of Sentiments in Dataset")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nVocabulary size:", len(vectorizer.get_feature_names_out()))

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("\nModel Evaluation Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Positive"])
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nNumber of Positive reviews:", sum(y_test == "Positive"))
print("Number of Negative reviews:", sum(y_test == "Negative"))

user_review = input("\nEnter your review: ")
cleaned = clean_text(user_review)
user_vec = vectorizer.transform([cleaned])
prediction = model.predict(user_vec)[0]
print("\nPredicted Sentiment:", prediction)

feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]

top_positive = np.argsort(coefficients)[-10:]
top_negative = np.argsort(coefficients)[:10]

print("\nTop words for Positive:")
print(feature_names[top_positive])

print("\nTop words for Negative:")
print(feature_names[top_negative])
