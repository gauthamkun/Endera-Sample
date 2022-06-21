#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

import numpy as np
import pandas as pd
import spacy
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL
from spacy.training import Example
from spacy.util import minibatch, compounding

with open(r"C:\Users\Gkrishna\Downloads\Test.json") as json_data:
    d = json.load(json_data)

df = pd.DataFrame(columns=['annotations', 'label'])
# displaying the DataFrame
i = 0
tags = []
loop = {}
x = []
y = []
z = []

df = pd.DataFrame(columns=tags)

for text, annotations in d["annotations"]:  # to loop through the text , annotations
    if len(text) > 0:  # checks for empty cases
        i = i + 1
        df.loc[i, ['annotations']] = [text]
        x.append(text)

        for start, end, label in annotations['entities']:
            df.loc[i, ['label']] = [label]
            df.loc[i, [label]] = 1
            int(df.loc[i, [label]])
            y.append(label)

for i in range(len(x)):
    z.append((x[i], {"cats": y[i]}))
df.replace(np.NaN, 0, inplace=True)
del df['label']

for label in d["classes"]:
    df[label] = df[label].apply(np.int64)
df.dtypes

# In[2]:


df.head(3)

# In[3]:


nlp = spacy.blank("en")

config = {
    "threshold": 0.5,
    "model": DEFAULT_MULTI_TEXTCAT_MODEL,
}

textcat = nlp.add_pipe(
    "textcat_multilabel",
    config=config)

# In[4]:


for label in d["classes"]:
    textcat.add_label(label)
textcat.labels


# In[5]:


def process_text(df, tags):
    texts = df.annotations

    labels = []
    for row in range(len(df)):
        label_dict = dict()
        for tag in tags:
            label_dict[tag] = df.iloc[row, df.columns.get_loc(tag)] == 1
        labels += [{'cats': label_dict}]

    return texts, labels


# In[6]:


process_text(df, textcat.labels)[0]

# In[7]:


train_data = list(zip(*process_text(df, textcat.labels)))

# In[8]:


train_examples = [Example.from_dict(nlp.make_doc(text), label)
                  for text, label in train_data]
textcat.initialize(lambda: train_examples, nlp=nlp)
train_examples

epochs = 20
with nlp.select_pipes(enable="textcat_multilabel"):
    optimizer = nlp.resume_training()
    batches = minibatch(train_data, size=compounding(4., 32., 1.001))
    for i in range(epochs):
        losses = {}
        for batch in batches:
            for text, label in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, label)
                print(nlp.update([example], sgd=optimizer, drop=0.2, losses=losses))
