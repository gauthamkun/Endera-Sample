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
    with open(r"C:\Users\Gkrishna\Downloads\Gautham_KY_1-2030.json") as json_data:
        d = json.load(json_data)
    df = pd.DataFrame(columns=['annotations', 'label'])  # Create a dataframe with headers annotations and label
    i = 0
    x = []
    y = []
    z = []
    print("Starting data preprocessing")
    for text, annotations in d["annotations"]:
        if len(text) > 0:
            i = i + 1
            df.loc[i, ['annotations']] = [text]
            x.append(text)  # Stores the annotations to the dataframe
            for start, end, label in annotations['entities']:
                df.loc[i, ['label']] = [label]
                df.loc[i, [label]] = 1  # Stores 1 for the label
                int(df.loc[i, [label]])
                y.append(label)

    for i in range(len(x)):
        z.append((x[i], {"cats": y[i]}))
    df.replace(np.NaN, 0, inplace=True)
    del df['label']

    for label in d["classes"]:
        df[label] = df[label].apply(np.int64)  # It usually gets stored as float this converts to Int
    print(df)
    print("Done with data preprocessing")
    return df, d


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

    for label in d["classes"]:  # add all the labels to model
        textcat.add_label(label)
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
