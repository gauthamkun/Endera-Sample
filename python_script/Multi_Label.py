import json
import numpy as np
import pandas as pd
import spacy
import random
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL
from spacy.training import Example
from spacy.util import minibatch, compounding


# used of converting the json data into dataframe
def preprocess_data():
    train = pd.read_csv('Data.csv')
    column_name = []
    counter = 0
    for col in train.columns:
        if counter != 0 and counter != 1:
            column_name.append(col)
        counter = counter+1
    return train, column_name

# for creating pipeline and model
def create_pipeline(d):
    print("Creating pipeline")
    nlp = spacy.load("en_core_web_md")  # using preexisting models
    config = {
        "threshold": 0.5,  # Cutoff to consider a prediction “positive”, relevant when printing accuracy results
        "model": DEFAULT_MULTI_TEXTCAT_MODEL,  # This model needs to be used for anything for more than 2 label
    }
    textcat = nlp.add_pipe("textcat_multilabel", config=config, last=True)
    print("Created pipeline")

    for label in d:  # add all the labels to model
        textcat.add_label(label)
    print(textcat.labels)
    return textcat, nlp


# used for converting the the dataframe to the format the model uses
def process_text(df, tags):
    texts = df.annotations

    labels = []
    for row in range(len(df)):
        label_dict = dict()
        for tag in tags:
            label_dict[tag] = df.iloc[row, df.columns.get_loc(tag)] == 1
        labels += [{'cats': label_dict}]

    return texts, labels


# The model is trained for the data provided
def training(textcat, train, nlp):
    train_data = list(zip(*process_text(train, textcat.labels)))
    print("Training the data will take time")
    train_examples = [Example.from_dict(nlp.make_doc(text), label)  # tag per token
                      for text, label in train_data]
    textcat.initialize(lambda: train_examples, nlp=nlp)
    epochs = 10
    with nlp.select_pipes(enable="textcat_multilabel"):
        optimizer = nlp.resume_training()
        batches = minibatch(train_data, size=compounding(4., 32., 1.001))  # infinite series start, stop, compound
        for i in range(epochs):
            random.shuffle(train_data)
            for batch in batches:
                for text, label in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, label)
                    print(nlp.update([example], sgd=optimizer))
    # save the trained model so that it doesnt have to be executed for input.
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("C:/Users/Gkrishna/OneDrive - Endera Systems/Documents/Python Scripts/python_stuff_new")
    print("Saved model")


def main():
    train, data = preprocess_data()
    textcat, nlp = create_pipeline(data)
    training(textcat, train, nlp)


if __name__ == "__main__":
    main()
