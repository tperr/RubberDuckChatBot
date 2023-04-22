import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Model, load_model
from os.path import exists
import time
import json
from bs4 import BeautifulSoup

"""
Division of labor:
Tim Perr: get_model, html removing
Ransom Duncan: load_questions, load_answers
Josh Farr: get_tokenizer, anything not in functions
Quentin Ross: get_answer, classify_question, training the model
"""

# loads questions from specified filename, usually "Questions.csv"
def load_questions(filename):
    questions_df = pd.read_csv(filename, encoding="iso-8859-1")  # why will UTF-8 not work??????
    questions_df = questions_df.iloc[:, [0, 1, 2, 3, 4, 5]]
    data = questions_df["Title"].values.tolist()
    data = [BeautifulSoup(value, "html.parser").get_text() for value in data]
    labels = questions_df.iloc[:, 0].astype(str).tolist()

    categories = set(labels)
    return data, categories


# gets the tokenizer, whether that be creating it if it doesn't exist or loading it from the specified file
def get_tokenizer(data, filename):
    if not exists(filename):
        # tokenize data
        print("Tokenizing questions...")
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(row for row in data)
        sequences = tokenizer.texts_to_sequences(row for row in data)
        length = max([len(seq) for seq in sequences])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=length)

        np.savez(filename, tokenizer=tokenizer.to_json(), padded_sequences=padded_sequences)
        
    else:
        # load tokenized data from file, makes it faster by like a minute and a half
        print("Loading tokenized questions...")

        data = np.load(filename, allow_pickle=True)

        # this part was an absolute pain to make work
        tokenizer_data = data["tokenizer"]
        tokenizer_data_string = json.dumps(tokenizer_data.tolist())
        tokenizer_config = json.loads(tokenizer_data_string)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

        padded_sequences = data["padded_sequences"]
        length = padded_sequences.shape[1]
    return tokenizer, padded_sequences, length


# loads the answers from Answers.csv
def load_answers(filename):
    print("Loading answers...")
    with open(filename, encoding="iso-8859-1") as file:
        data = list(csv.reader(file))[1:]  # skip header row
        return [BeautifulSoup(row[5], "html.parser").get_text() for row in data if row[5]]  # unhtmls answer body
    

# gets model, either trains it or creates a new one
def get_model(input_length, tokenizer, categories, sequences):
    train_more = False
    if exists("rubbermodel.h5"):
        print("Loading model...")
        model = load_model("rubbermodel.h5")
        print("Model loaded...")
        return model
    else:
        with tf.device("/GPU:0"):
            print("Creating model...")
            # create model
            input_layer = Input(shape=(input_length))
            embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32)(input_layer)
            conv_layer = Conv1D(filters=32, kernel_size=2, padding="same", activation="relu")(embedding_layer)

            pooling_layer = GlobalMaxPooling1D()(conv_layer)

            output_layer = Dense(len(categories), activation="softmax")(pooling_layer)
            model = Model(inputs=input_layer, outputs=output_layer)

            model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

            category_encoder = {category: i for i, category in enumerate(categories)}
            encoded = [category_encoder[category] for category in categories]
            encoded = np.array(encoded)

            sequences = np.array(sequences)

            x = tf.convert_to_tensor(sequences, dtype=tf.float32)

            y = tf.convert_to_tensor(encoded, dtype=tf.float32)

            # train model, increase batch size if you want
            print("Fitting model...")

            model.fit(x=x, y=y, epochs=10, batch_size=32)
            model.save("rubbermodel.h5")
            print("Model trained successfully...")

    if train_more:
        # just trains more if wanted
        with tf.device("/GPU:0"):
            print("training more")
            model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            categories = np.array(categories)
            sequences = np.array(sequences)

            x = tf.convert_to_tensor(sequences, dtype=tf.float32)

            y = tf.convert_to_tensor(categories, dtype=tf.float32)
            model.fit(x=x, y=y, epochs=10, batch_size=32)
            model.save("rubbermodel.h5")


# gets answer based off the category
def get_answer(category, answers):
    
    #TODO make this so it takes all the categories and gets the answer with the most categories
    relevant_answers = [answer for answer in answers if category.lower() in answer.lower()]
    if relevant_answers:
        return relevant_answers[0]
    else:
        return "Sorry, I don't know the answer to that question."


# classifies question
def classify_question(query, tokenizer, length, categories, model, answers):
    print("Classifying Question: '", query + "'")
    # process query
    query_sequence = tokenizer.texts_to_sequences([query])
    query_padded = tf.keras.preprocessing.sequence.pad_sequences(query_sequence, maxlen=length)

    # classify query
    category_idx = np.argmax(model.predict(query_padded))
    category = list(categories)[category_idx]

    # get relevant answer
    answer = get_answer(category, answers)

    return answer


totalTime = time.time()

print("Loading questions...")
startTime = time.time()

data, categories = load_questions("Questions.csv") 

print("Loading questions took", time.time() - startTime, "seconds")
startTime = time.time()

tokenizer, paddedSequences, length = get_tokenizer(data, "tokenized_questions.npz")

print("Tokenizing took", time.time() - startTime, "seconds")
startTime = time.time()

answers = load_answers("Answers.csv")

print("Loading answers took", time.time() - startTime, "seconds")

print("Total preprocessing took ", time.time() - totalTime, "seconds")
model = get_model(length, tokenizer, categories, paddedSequences)
print("Total time took", time.time() - totalTime, "seconds")

question = input("Enter a question: ")
# test it
while question != "quit":
    print(classify_question(question, tokenizer, length, categories, model, answers))
    question = input("Enter a question: ")
