german_to_english = {
    "hallo": "hello",
    "hi": "hi",
    "guten": "good",
    "morgen": "morning",
    "tag": "day",
    "abend": "evening",
    "nacht": "night",
    "willkommen": "welcome",
    "tschüss": "bye",
    "auf": "on",
    "wiedersehen": "goodbye"
}

def clean_text(sentence):
    cleaned = ""
    for ch in sentence:
        if ch.isalpha() or ch.isspace():
            cleaned += ch
        else:
            cleaned += " "
    return cleaned.strip().lower()

def tokenize(text):
    return [word for word in text.split() if word]

def translate_tokens(tokens):
    translated = []
    for word in tokens:
        if word in german_to_english:
            translated.append(german_to_english[word])
        else:
            translated.append(word)
    return translated

def reconstruct_sentence(translated_tokens):
    if not translated_tokens:
        return ""
    translated_tokens[0] = translated_tokens[0].capitalize()
    return " ".join(translated_tokens)

def translate_sentence(sentence):
    cleaned = clean_text(sentence)
    tokens = tokenize(cleaned)
    translated_tokens = translate_tokens(tokens)
    result = reconstruct_sentence(translated_tokens)
    return result


print("Simple German → English Translator (Built-in Python Only)")
print("Type 'exit' to quit.\n")

while True:
    german_sentence = input("Enter a German sentence: ").strip()
    if german_sentence.lower() == "exit":
        print("Goodbye!")
        break
    english_translation = translate_sentence(german_sentence)
    print("English translation:", english_translation, "\n")
    
""" 
# pip install googletrans==4.0.0rc1

from googletrans import Translator

translator = Translator()

source_text = input("Enter text to translate: ")
source_lang = input("Enter source language code (example: en): ")
dest_lang = input("Enter destination language code (example: fr): ")

translated = translator.translate(
    source_text,
    src=source_lang,
    dest=dest_lang
)

print("Translated text:", translated.text)
 """