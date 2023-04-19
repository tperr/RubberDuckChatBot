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

print("Loading tags...")
# load and preprocess tags
tagsDataframe = pd.read_csv("Tags.csv")
tagList = tagsDataframe['Tag'].str.split().tolist() 

if not exists("tokenizedTags.npz"):
    print("Tokenizing tags...")
    # tokenizing tags
    tagTokenizer = tf.keras.preprocessing.text.Tokenizer()
    tagTokenizer.fit_on_texts([str(tag) for tag in tagList])
    tagSequences = tagTokenizer.texts_to_sequences([str(tag) for tag in tagList])
    tagLength = max([len(seq) for seq in tagSequences])
    paddedTagSequences = tf.keras.preprocessing.sequence.pad_sequences(tagSequences, maxlen=tagLength)
    np.savez("tokenizedTags.npz", tokenizer=tagTokenizer.to_json(), paddedSequences=paddedTagSequences)
else:
    print("Loading tokenized tags...")
    
    data = np.load('tokenizedTags.npz', allow_pickle=True)

    # this part was an absolute pain to make work
    tokenizerData = data['tokenizer']
    tokenizerDataString = json.dumps(tokenizerData.tolist())
    tokenizerConfig = json.loads(tokenizerDataString)
    tagTokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizerConfig)

    paddedTagSequences = data['paddedSequences']
    tagLength = paddedTagSequences.shape[1]


print("Tag loading took ", time.time() - startTime, "seconds")
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
    reshapeLayer = Reshape((1, 128))(poolingLayer) # needed so that the tag layer and the sequences are same shape
    flattenLayer = Flatten()(reshapeLayer)

    tagInputLayer = Input(shape=(tagLength,))
    tagEmbeddingLayer = Embedding(input_dim=len(tagTokenizer.word_index) + 1, output_dim=128)(tagInputLayer)
    tagFlattenLayer = Flatten()(tagEmbeddingLayer)

    # concatenate question and tag embeddings
    concatLayer = Concatenate()([poolingLayer, tagFlattenLayer])

    outputLayer = Dense(len(categories), activation="softmax")(concatLayer)
    model = Model(inputs=[inputLayer, tagInputLayer], outputs=outputLayer)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # convert labels to integers
    labelEncoder = {category: i for i, category in enumerate(categories)}
    labels = [labelEncoder[label] for label in labels]
    labels = np.array(labels)

    # train model, increase batch size if you want
    model.fit(x=np.array([np.array(paddedSequences), np.array(paddedTagSequences)]), y=labels, batch_size=32, epochs=10)
    model.save("rubbermodel.h5")
    
print("Model trained")
print("Total time took", time.time() - totalTime, "seconds")
exit()
def classifyQuestion(query, tokenizer, tagTokenizer, length, categories, model):
    # process it
    querySequence = tokenizer.texts_to_sequences([query])
    queryPadded = tf.keras.preprocessing.sequence.pad_sequences(querySequence, maxlen=length)

    # classify it
    categoryIndex = np.argmax(model.predict([queryPadded, np.zeros((1, tagLength))]), axis=-1)[0]
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