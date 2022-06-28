from unittest import TestCase
from pandas.testing import assert_frame_equal
from Multi_Label import preprocess_data
import numpy as np
import pandas as pd
import json


class Test(TestCase):
    with open(r"C:\Users\mandr\Desktop\Internship\Week 1\0-1000\andrew-ky_offense_vocabulary-0_1000.json") as json_data:
        d = json.load(json_data)
    df = pd.DataFrame(columns=['annotations', 'label'])
    i = 0
    x = []
    y = []
    z = []
    for text, annotations in d["annotations"]:
        if len(text) > 0:
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

    def test_preprocess_data_dataframe(self):
        output, data = preprocess_data()
        assert len(output) != 0

    def test_preprocess_data_json_data(self):
        output, data = preprocess_data()
        assert len(data) != 0

    def test_preprocess_data_dataframe_gives_right_data(self):
        output, data = preprocess_data()
        assert_frame_equal(output, Test.df)
