# pip install tensorflow


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore  
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt
import pickle
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

file_path = "./datasets/Sherlock Holmes.txt"
with open(file_path, "r", encoding="utf8") as f:
    text = f.read()

text = text.lower()
text = re.sub(r"[^a-zA-Z\s]", "", text)
text = re.sub(r"\s+", " ", text).strip()

print(f" Loaded text with {len(text.split())} words.")

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

pickle.dump(tokenizer, open("token.pkl", "wb"))

total_words = len(tokenizer.word_index) + 1
print(f" Total unique words in dataset: {total_words}")

input_sequences = []
token_list = tokenizer.texts_to_sequences([text])[0]

for i in range(4, len(token_list)):
    seq = token_list[i - 4 : i + 1]
    input_sequences.append(seq)

max_seq_len = max(len(x) for x in input_sequences)
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_seq_len, padding="pre")
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

print(f" Total training samples: {X.shape[0]} | Sequence length: {max_seq_len}")

model = Sequential(
    [
        Embedding(total_words, 64, input_length=max_seq_len - 1),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(256, activation="relu"),
        Dense(total_words, activation="softmax"),
    ]
)

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.summary()

checkpoint = ModelCheckpoint(
    "best_next_word_model.h5", monitor="loss", save_best_only=True, verbose=1
)
early_stop = EarlyStopping(
    monitor="loss", patience=3, restore_best_weights=True
)

history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=128,
    callbacks=[checkpoint, early_stop],
)

plt.plot(history.history["loss"], label="Loss")
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

model = load_model("best_next_word_model.h5")
print(" Model loaded successfully!")

def predict_next_word(model, tokenizer, seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_seq_len - 1, padding="pre"
    )
    predicted = np.argmax(model.predict(token_list), axis=-1)[0]
    predicted_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            predicted_word = word
            break
    return predicted_word

print("\n LSTM Text Generator Ready!")
print("Type a short phrase (at least 3 words). Type '1' to exit.\n")

while True:
    user_input = input("Enter your line: ").strip()
    if user_input == "1":
        print("\n Exiting program. Goodbye!")
        break
    if len(user_input.split()) < 3:
        print(" Please enter at least 3 words.\n")
        continue
    next_word = predict_next_word(model, tokenizer, user_input)
    print(f"\n Predicted next word: {next_word}\n")
