import spacy
import operator
import json

def funs():
    texts = "AGGRAVATED ASSAULT - FAMILY OR HOUSEHOLD MEMBER"  # sample data
    texts1 = "POSSESSION OF DRUG PARAPHERNALIA"
    texts2 = "CT- DRIVING UNDER THE INFLUENCE CAR"

    # call the model
    nlp = spacy.load("C:/Users/mandr/Desktop/Internship/Python Scripts/python_stuff_new")

    with open(r"C:\Users\mandr\Desktop\Internship\Week 1\0-1000\andrew-ky_offense_vocabulary-0_1000.json") as json_data:
        d = json.load(json_data)

    tags_true = [annotations[1]['entities'][0][2] for annotations in d['annotations']] #creates a list of the 'true' tags
    text_offenses = [annotations[0] for annotations in d['annotations']] #creates a list of the offenses
    tags_pred = [annotations[1]['entities'][0][2] for annotations in d['annotations']] #creates a list to store the tags, will be overwritten by the predicted ones


    counter = 0 #creates a counter at 0

    for str in text_offenses: #iterates through the offense strings

    #can find number of annotations using the json beautifier https://codebeautify.org/jsonviewer
        if (counter < 201): #will need to change 201 to however many samples you are using, fix this

            text = text_offenses[counter] #sets the offense to be processed to whatever index the counter is at

            doc = nlp(text)# to create a doc spacy objecy

            # gives the score prediction for the model
            article_categories = doc.cats
            # arranges them in descending order

            #get the first one, ask Gautham tomorrow
            for k, v in sorted(article_categories.items(), key=operator.itemgetter(1), reverse=True):
                tags_pred[counter] = k
                break

            counter = counter + 1 #adds 1 to the counter

        else:
            break #once it has reached the end of the list, the loop breaks

    from sklearn.metrics import confusion_matrix #imports the confusion matrix from scikit learn
    #inputs the true tags and predicted tags into the confusion matrix

    print("Tags, left to right, top to bottom")
    print(d['classes'])
    print(confusion_matrix(tags_true, tags_pred, labels = ['TRESSPASS', 'FRAUD', 'WEAPON', 'TRAFFIC', 'ASSAULT', 'OBSTRUCTION', 'FINANCIAL']))

def main():
    funs()


if __name__ == "__main__":
    main()
