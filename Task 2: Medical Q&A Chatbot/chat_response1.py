import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load model and resources
model = load_model("chatbot_model_new.keras")
with open("words.pkl", "rb") as f:
    words = pickle.load(f)
with open("classes.pkl", "rb") as f:
    classes = pickle.load(f)

# Load CSV
df = pd.read_csv("Extracted_data_From_XML.csv")
df["tag"] = df["Focus"].fillna("general").str.lower()  # SAME as training!

def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.isalnum()]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intent, df):
    responses = df[df["tag"] == intent]["Answer"].tolist()
    return random.choice(responses) if responses else "Sorry, I don't have an answer for that."

def chatbot_response(text):
    intents = predict_class(text)
    if intents:
        return get_response(intents[0]["intent"], df)
    else:
        return "Sorry, I couldn't understand that."

# Optional: quick test
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        print("Bot:", chatbot_response(user_input))
