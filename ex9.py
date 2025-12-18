from sentence_transformers import SentenceTransformer, util
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import random
import torch
import json
import nltk
import re

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

print("Loading model... (please wait)")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading intents dataset...")
with open("./datasets/clg_intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

texts = []
responses = []
tags = []

for intent in data["intents"]:
    tag = intent["tag"] if "tag" in intent else "general"
    for question in intent["patterns"]:
        cleaned_q = clean_text(question)
        texts.append(cleaned_q)
        responses.append(random.choice(intent["responses"]))
        tags.append(tag)

print(f"Total patterns loaded: {len(texts)}")

print("Encoding all possible questions...")
text_embeddings = model.encode(texts, convert_to_tensor=True)

interaction_log = []
topic_counter = Counter()

def chatbot_response(user_input):
    cleaned_input = clean_text(user_input)
    user_embedding = model.encode(cleaned_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, text_embeddings)
    best_match_idx = torch.argmax(similarities)
    best_score = similarities[0][best_match_idx].item()

    if best_score < 0.45:
        return "I'm not sure I understand your question. Could you try rephrasing it?", None, best_score

    response = responses[best_match_idx]
    tag = tags[best_match_idx]

    clean_response = re.sub(r"<.*?>", "", response).strip()

    if re.search(r"\b(visit|website|college|portal)\b", clean_response, re.IGNORECASE):
        clean_response += " You can also check: https://sairam.edu.in/"

    topic_counter[tag] += 1
    return clean_response, tag, best_score

print("\nCollege Enquiry Bot: Hello! I'm your University Assistant.")
print("Ask me about admissions, departments, or college facilities.")
print("Type 'quit' to end the chat.\n")

conversation_count = 0

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("\nCollege Enquiry Bot: Goodbye! Have a great day ahead!")
        break

    reply, tag, score = chatbot_response(user_input)
    print(f"College Enquiry Bot: {reply}\n")

    interaction_log.append(
        {
            "user": user_input,
            "bot": reply,
            "confidence": round(score, 3) if score else 0,
        }
    )

    conversation_count += 1

print("\n=== Chat Summary ===")
print(f"Total user messages: {conversation_count}")

if topic_counter:
    print("\nMost discussed topics:")
    for topic, count in topic_counter.most_common(3):
        print(f"• {topic.capitalize()} ({count} times)")

with open("./datasets/university_chat_log.json", "w", encoding="utf-8") as f:
    json.dump(interaction_log, f, indent=2)

print("\nChat history saved to 'university_chat_log.json'")
print("Thanks for chatting with the University Bot!")
