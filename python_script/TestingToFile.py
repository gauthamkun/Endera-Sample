import spacy
import pandas as pd


def funs():
    # Change text inside parenthesis to be the pathway of wherever you saved the model. (for most of you it will be the folder named python_stuff_new)
    nlp = spacy.load('C:/Users/jcosta/Desktop/Endera-Sample/Trained_model')
    # Give the input file (give pathway to the file you'd like to test)
    df = pd.read_fwf('C:/Users/jcosta/Desktop/Endera-Sample/Datasets/Test.txt', header=None, names=['Annotations', 'MISCONDUCT',
                                                                                                           'ALCOHOL', 'VEHICULAR',
                                                                                                           'MINOR', 'SEX', 'KIDNAP', 'VIOLENCE',
                                                                                                           'WEAPON',
                                                                                                           'THEFT', 'FRAUD',
                                                                                                           'TRESPASS',
                                                                                                           'DRUG', 'COURT OFFENSE',
                                                                                                           'LICENSE','HARASSMENT', 'ACCESSORY', 'ANIMAL'])
    data = df['Annotations']
    docs1 = list(nlp.pipe(data))
    i=0
    for doc in docs1:
        article_categories = doc.cats
        for k, v in article_categories.items():
            df.loc[i, [k]] = v


        i = i+1
    # Give the output path (Give pathway of wherever you'd like to save the csv export. Be sure to rename your export every time or it will overwrite the old one)
    # File automatically gets created
    df.to_csv('C:/Users/jcosta/Desktop/Endera-Sample/csv exports/Output1.csv', encoding='utf-8')
    print("Data saved to csv file")


def main():
    funs()


if __name__ == '__main__':
    main()
