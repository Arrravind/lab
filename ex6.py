# pip install wordcloud vaderSentiment


import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

print("\n=== WHATSAPP CHAT ANALYSIS USING NLP ===\n")

file_path = "./datasets/WhatsApp Chat with SECAD2027B.txt"
with open(file_path, "r", encoding="utf-8") as f:
    chat = f.readlines()

dates = []
users = []
messages = []
hours = []

pattern = r"(\d+/\d+/\d+),\s(\d+):(\d+)\s?(am|pm)?\s-\s([^:]+):\s(.*)"

for line in chat:
    match = re.match(pattern, line, flags=re.IGNORECASE)
    if not match:
        continue

    date, hour, minute, ampm, user, msg = match.groups()

    hour = int(hour)
    if ampm:
        if ampm.lower() == "pm" and hour != 12:
            hour += 12
        elif ampm.lower() == "am" and hour == 12:
            hour = 0

    msg = re.sub(r"(http\S+|<Media omitted>|@\d+)", "", msg).strip()
    if not msg:
        continue

    users.append(user)
    messages.append(msg)
    hours.append(hour)
    dates.append(date)

if not messages:
    print("No valid messages found. Check file format (must be exported as .txt).")
else:
    print(f"\nTotal messages processed: {len(messages)}")
    print(f"Unique users detected: {len(set(users))}")

df = pd.DataFrame(
    {
        "Date": dates,
        "User": users,
        "Hour": hours,
        "Message": messages,
    }
)

top_user = df["User"].value_counts().idxmax()
print(f"Most active user: {top_user}")

avg_msgs = df.groupby("Date")["Message"].count().mean()
print(f"Average messages per day: {avg_msgs:.1f}")

all_text = " ".join(df["Message"]).lower()
words = re.findall(r"\b[a-zA-Z]+\b", all_text)
stopwords = {
    "the", "is", "and", "a", "to", "in", "of", "you", "i", "me", "it",
    "for", "on", "this", "that", "was", "we", "at"
}
clean_words = [w for w in words if w not in stopwords]

top_words = Counter(clean_words).most_common(10)
print("\nTop 10 Words:\n", top_words)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(" ".join(clean_words))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Chat Messages")
plt.show()

analyzer = SentimentIntensityAnalyzer()
sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}

for msg in df["Message"]:
    score = analyzer.polarity_scores(msg)["compound"]
    if score > 0.05:
        sentiments["Positive"] += 1
    elif score < -0.05:
        sentiments["Negative"] += 1
    else:
        sentiments["Neutral"] += 1

print("\nSentiment Summary:", sentiments)

plt.figure(figsize=(5, 5))
plt.pie(
    sentiments.values(),
    labels=sentiments.keys(),
    autopct="%1.1f%%",
    startangle=90,
)
plt.title("Sentiment Distribution")
plt.show()

hour_count = Counter(df["Hour"])
plt.figure(figsize=(8, 4))
plt.bar(hour_count.keys(), hour_count.values())
plt.title("Messages by Hour of Day")
plt.xlabel("Hour (0–23)")
plt.ylabel("Message Count")
plt.grid(axis="y")
plt.show()

user_count = df["User"].value_counts().head(5)
plt.figure(figsize=(16, 4))
plt.barh(user_count.index, user_count.values)
plt.title("Top 5 Active Users")
plt.xlabel("Message Count")
plt.ylabel("User")
plt.gca().invert_yaxis()
plt.show()

print("\nWhatsApp Chat Analysis completed successfully.\n")
