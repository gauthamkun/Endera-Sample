import numpy as np
import pandas as pd
import json
from itertools import chain
import itertools
from sqlalchemy import create_engine


def connection1():
    db_connection_str = 'mysql+pymysql://root:1234@127.0.0.1/make_insights'
    db_connection = create_engine(db_connection_str)
    df = pd.read_sql(
        'SELECT CONVERT(conclusion_file using utf8) as data FROM make_insights.insight where analyst_id != 14',
        con=db_connection)
    len1 = len(df)
    labels = []
    anns = []
    test = []
    i1 = 0
    x1 = []
    w1 = []
    z1 = []
    counter = 0
    counter1 = 0
    df1 = pd.DataFrame(columns=['annotations', 'label'])
    print(len1)
    for x in range(len1):
        if x != 40:
            s1 = df['data'][x]
            y2 = json.loads(s1)
            labels.append(y2["classes"])
            anns.append(y2["annotations"])
            test.append(y2)
    d = (set(chain(*labels)))
    listAnno = list(itertools.chain.from_iterable(anns))
    for text, annotations in listAnno:
        counter = counter + 1
        if len(annotations['entities']) > 0:
            i1 = i1 + 1
            df1.loc[i1, ['annotations']] = [text]
            x1.append(text)

            for start, end, labels in annotations['entities']:
                counter1 = counter1 + 1
                df1.loc[i1, ['label']] = [labels]
                df1.loc[i1, [labels]] = 1  # Stores 1 for the label
                int(df1.loc[i1, [labels]])
                w1.append(labels)

    for i in range(len(df1)):
        z1.append((x1[i], {"cats": w1[i]}))

    df1.replace(np.NaN, 0, inplace=True)
    del df1['label']

    for label in d:
        df1[label] = df1[label].apply(np.int64)  # It usually gets stored as float this converts to Int
    print(df1)
    df1.to_csv('C:/Users/Gkrishna/OneDrive - Endera Systems/Desktop/python_script/Data.csv', encoding='utf-8')
    print("Done with data preprocessing")


def main():
    connection1()


if __name__ == "__main__":
    main()
