import pandas as pd
import numpy as np
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load CSV
df = pd.read_csv("Extracted_data_From_XML_1.csv")
df["tag"] = df["Focus"].fillna("general").str.lower()

documents = []
words = []
classes = []

# Tokenize and prepare training data
for index, row in df.iterrows():
    question = row["Question"]
    tag = row["tag"]
    tokens = nltk.word_tokenize(str(question))
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens if w.isalnum()]
    words.extend(tokens)
    documents.append((tokens, tag))
    if tag not in classes:
        classes.append(tag)

words = sorted(set(words))
classes = sorted(set(classes))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]
    output_row = output_empty[:]
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

X = np.array(list(training[:, 0]))
y = np.array(list(training[:, 1]))

# Build and train model
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=8, verbose=1)

# Save model and helper files
model.save("chatbot_model_new.keras")
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
