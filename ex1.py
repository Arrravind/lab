# In Terminal :
# pip install nltk spacy
# python -m spacy download en_core_web_sm


import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
import spacy

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

user_text = input("Enter a text to tokenize: ")

tokens = word_tokenize(user_text)
print("\nOriginal text:", user_text)
print("Word tokens:", tokens)

tweet_tokenizer = TweetTokenizer()
tweet_text = input("\nEnter a tweet text to tokenize: ")
tweet_tokens = tweet_tokenizer.tokenize(tweet_text)
print("\nOriginal tweet:", tweet_text)
print("Tweet tokens:", tweet_tokens)

stop_words = set(stopwords.words("english"))
clean_tokens = [token for token in tokens if token.lower() not in stop_words]
clean_text = " ".join(clean_tokens)
print("\nText after stopwords removal:", clean_text)

nlp = spacy.load("en_core_web_sm")
doc = nlp(user_text)
spacy_tokens = [token.text for token in doc]
print("\nspaCy tokens:", spacy_tokens)

result = [(token.lemma_, token.pos_) for token in doc]
print(result)

sentence_text = input("\nEnter a text for sentence segmentation: ")
doc = nlp(sentence_text)
sentences = [sentence.text for sentence in doc.sents]
print("\nSentences:", sentences)
