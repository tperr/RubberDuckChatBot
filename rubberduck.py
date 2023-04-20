import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Flatten, Reshape
from tensorflow.keras.models import Model, load_model
from os.path import exists
import time
import json


totalTime = time.time()

print("Loading questions...")
startTime = time.time()

questionsDataFrame = pd.read_csv("Questions.csv", encoding="iso-8859-1") # why will UTF-8 not work??????
questionsDataFrame = questionsDataFrame.iloc[:, [0, 1, 2, 3, 4, 5]]
data = questionsDataFrame[["Title", "Body"]].values.tolist()
labels = questionsDataFrame.iloc[:, 0].astype(str).tolist()

categories = set(labels)

print("Loading questions took", time.time() - startTime, "seconds")
startTime = time.time()

if not exists("tokenizedQuestions.npz"):
    # tokenize data
    print("Tokenizing questions...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(row[0] + " " + row[1] for row in data)
    sequences = tokenizer.texts_to_sequences(row[0] + " " + row[1] for row in data)
    length = max([len(seq) for seq in sequences])
    paddedSequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=length)
    
    np.savez("tokenizedQuestions.npz", tokenizer=tokenizer.to_json(), paddedSequences=paddedSequences)
else:
    # load tokenized data from file, makes it faster by like a minute and a half
    print("Loading tokenized questions...")
    
    data = np.load('tokenizedQuestions.npz', allow_pickle=True)

    # this part was an absolute pain to make work
    tokenizerData = data['tokenizer']
    tokenizerDataString = json.dumps(tokenizerData.tolist())
    tokenizerConfig = json.loads(tokenizerDataString)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizerConfig)

    paddedSequences = data['paddedSequences']
    length = paddedSequences.shape[1]

print("Tokenizing took", time.time() - startTime, "seconds")
startTime = time.time()

print("Total preprocessing took ", time.time() - totalTime, "seconds")

if exists("rubbermodel.h5"):
    print("Loading model...")
    model = load_model("rubbermodel.h5")
else:
    print("Creating model...")
    # create model
    inputLayer = Input(shape=(length,))
    embeddingLayer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(inputLayer)
    convLayer = Conv1D(filters=128, kernel_size=3, activation="relu")(embeddingLayer)
    poolingLayer = GlobalMaxPooling1D()(convLayer)

    outputLayer = Dense(len(categories), activation="softmax")(poolingLayer)
    model = Model(inputs=inputLayer, outputs=outputLayer)

    with tf.device('/GPU:0'):

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # convert labels to integers
    labelEncoder = {category: i for i, category in enumerate(categories)}
    labels = [labelEncoder[label] for label in labels]
    labels = np.array(labels)

    paddedSequences = np.array(paddedSequences)

    x = tf.convert_to_tensor(paddedSequences, dtype=tf.float32)

    y = tf.convert_to_tensor(labels, dtype=tf.float32)
    
    # train model, increase batch size if you want
    print("Fitting model...")
    with tf.device('/GPU:0'):
        model.fit(x=x, y=y, epochs=10, batch_size=32)

    model.save("rubbermodel.h5")
    
print("Model trained")
print("Total time took", time.time() - totalTime, "seconds")
exit()
def classifyQuestion(query, tokenizer, length, categories, model):
    # process it
    querySequence = tokenizer.texts_to_sequences([query])
    queryPadded = tf.keras.preprocessing.sequence.pad_sequences(querySequence, maxlen=length)

    # classify it
    categoryIndex = np.argmax(model.predict([queryPadded, np.zeros((1))]), axis=-1)[0]
    category = list(categories)[categoryIndex]

    # get best answer
    with open("answer.csv") as file:
        data = list(csv.reader(file))
    answers = [row[5] for row in data]  # Only keep answers with no parent question
    relevantAnswers = [answer for answer in answers if category.lower() in answer.lower()]
    if relevantAnswers:
        return relevantAnswers[0]
    else:
        return "Sorry, I don't know the answer to that question."


print("done")