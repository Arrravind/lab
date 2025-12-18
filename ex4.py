import math
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import matplotlib.pyplot as plt

nltk.download("punkt")
nltk.download("stopwords")

print("\n=== KEYWORD EXTRACTION USING NLP (TF-IDF BASED) ===\n")

doc = input("Enter your text paragraph:\n")

stop_words = set(stopwords.words("english"))
sentences = sent_tokenize(doc)
words = word_tokenize(doc.lower())
filtered_words = [w for w in words if w not in stop_words and w not in punctuation]

print(f"Total Sentences : {len(sentences)}")
print(f"Total Words : {len(words)}")
print(f"Filtered Words : {len(filtered_words)}")

tf = {}
for word in filtered_words:
    tf[word] = tf.get(word, 0) + 1

for word in tf:
    tf[word] = tf[word] / len(filtered_words)

def count_sentences_containing(word, sentences):
    return sum(1 for sent in sentences if word in sent.lower())

idf = {}
for word in tf:
    idf[word] = math.log((1 + len(sentences)) / (1 + count_sentences_containing(word, sentences))) + 1

tf_idf = {word: tf[word] * idf[word] for word in tf}

sorted_keywords = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:15]

df_keywords = pd.DataFrame(sorted_keywords, columns=["Keyword", "TF-IDF Score"])

print("\nTop Keywords and their TF-IDF Scores:\n")
print(df_keywords.to_string(index=False))

word_freq = pd.Series(tf).sort_values(ascending=False)[:10]

plt.figure(figsize=(16, 12))

plt.subplot(2, 1, 1)
plt.barh(word_freq.index, word_freq.values, color='green')
plt.gca().invert_yaxis()
plt.title("Top 10 Frequent Words (Before TF-IDF)")
plt.xlabel("Frequency")
plt.ylabel("Words")

plt.subplot(2, 1, 2)
plt.barh(df_keywords["Keyword"], df_keywords["TF-IDF Score"])
plt.gca().invert_yaxis()
plt.title("Top Keywords by TF-IDF Score")
plt.xlabel("TF-IDF Score")
plt.ylabel("Keywords")
plt.show()

print("\nKeyword Extraction using NLP completed successfully.\n")
