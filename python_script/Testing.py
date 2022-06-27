import spacy
import operator


def funs():
    texts = "AGGRAVATED ASSAULT - FAMILY OR HOUSEHOLD MEMBER"  # sample data
    texts1 = "POSSESSION OF DRUG PARAPHERNALIA"
    texts2 = "CT- DRIVING UNDER THE INFLUENCE CAR"
    print(texts2)
    # call the model
    nlp = spacy.load("C:/Users/Gkrishna/OneDrive - Endera Systems/Documents/Python Scripts/python_stuff_new")
    docs = nlp(texts2)
    # gives the score prediction for the model
    article_categories = docs.cats
    # arranges them in descending order
    for k, v in sorted(article_categories.items(), key=operator.itemgetter(1), reverse=True):
        print(k, ":", v)


def main():
    funs()


if __name__ == "__main__":
    main()
