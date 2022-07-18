import spacy
import pandas as pd


def funs():
    # Change this to the local path.
    # Has the path to the model
    nlp = spacy.load('C:/Users/Gkrishna/Downloads/Trained_model/Trained_model')
    # Give the input file
    df = pd.read_fwf("C:/Users/Gkrishna/Downloads/TestTxt.txt", header=None, names=['Annotations', 'MISCONDUCT'
        , 'ALCOHOL', 'VEHICULAR', 'FAMILY', 'MINOR', 'SEX', 'KIDNAP', 'VIOLENCE', 'DRUGS', 'WEAPON', 'FINANCIAL'
        , 'THEFT', 'FRAUD', 'STALK', 'TRESPASS', 'DRUG', 'BURGLARY', 'COURT OFFENSE', 'LICENSE', 'POLICE OFFENSE'
        , 'HARASSMENT', 'ACCESSORY', 'ANIMAL', 'VEHICHULAR', 'TRESSPASS', 'TRAFFIC', 'OBSTRUCTION', 'ASSAULT'])
    data = df['Annotations']
    docs1 = list(nlp.pipe(data))
    i = 0
    for doc in docs1:
        # doc.cats gives the confidence value or prediction within range from 0 to 1.
        # Sum of all the labels prediction will be 1
        article_categories = doc.cats
        for k, v in article_categories.items():
            df.loc[i, [k]] = v

        i = i + 1
    # Give the output path
    # File automatically gets created
    df.to_csv('C:/Users/Gkrishna/OneDrive - Endera Systems/Desktop/python_script/Output4.csv', encoding='utf-8')
    print("Data saved to csv file ")


def main():
    funs()


if __name__ == "__main__":
    main()
