import json
import random
import string
import datetime
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

with open("./datasets/intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

patterns = []
tags = []
responses = {}

for intent in data["intents"]:
    tag = intent["tag"]
    responses[tag] = intent["responses"]
    for pattern in intent["patterns"]:
        patterns.append(clean_text(pattern))
        tags.append(tag)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

def get_response(user_input):
    cleaned_input = clean_text(user_input)
    user_vec = vectorizer.transform([cleaned_input])
    sim = cosine_similarity(user_vec, X)
    idx = sim.argmax()
    confidence = sim[0][idx]
    if confidence < 0.2:
        return "I'm not sure I understand. Could you please rephrase?", confidence
    tag = tags[idx]
    reply = random.choice(responses[tag])
    return reply, confidence

chat_history = []

def log_message(sender, message):
    chat_history.append(
        {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "sender": sender,
            "message": message,
        }
    )

print("ChatBuddy: Hello! I’m ChatBuddy, your NLP assistant.")
print("Type 'quit' to end the chat.\n")

message_count = 0

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("\nChatBuddy: Goodbye! Have a great day!")
        break

    log_message("User", user_input)
    reply, confidence = get_response(user_input)
    log_message("ChatBuddy", reply)
    print(f"ChatBuddy: {reply}\n")
    message_count += 1

with open("./datasets/chat_log.txt", "w", encoding="utf-8") as f:
    for msg in chat_history:
        f.write(f"[{msg['time']}] {msg['sender']}: {msg['message']}\n")

print(f"\nTotal messages exchanged: {message_count}")
print("Chat history saved successfully to 'chat_log.txt'.")
