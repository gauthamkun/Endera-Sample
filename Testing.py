# import statements
import spacy
import operator
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def funs():
    # call the model
    nlp = spacy.load("C:/Users/mandr/Desktop/Internship/Python Scripts/python_stuff_new")

    # opens the file used for testing
    with open(r"C:\Users\mandr\Desktop\Internship\Week 1\0-1000\andrew-ky_offense_vocabulary-0_1000.json") as json_data:
        d = json.load(json_data)

    # creates a list of the 'true' tags
    tags_true = [annotations[1]['entities'][0][2] for annotations in d['annotations']]

    # creates a list of the offenses
    text_offenses = [annotations[0] for annotations in d['annotations']]

    # creates a list to store the tags, will be overwritten by the predicted ones
    tags_pred = [annotations[1]['entities'][0][2] for annotations in d['annotations']]

    # creates a counter at 0
    counter = 0

    # iterates through the offense strings
    for str in text_offenses:

        # loops until the counter reaches the end of the offenses
        if counter < len(text_offenses):

            # sets the offense to be processed to whatever index the counter is at
            text = text_offenses[counter]

            # creates a doc spacy object
            doc = nlp(text)

            # gives the score prediction for the model
            article_categories = doc.cats

            # original model looped through all tags, I modified it to only take the greatest one (most likely tag)
            # will ask Gautham on how to make it so a for loop isn't needed
            for k, v in sorted(article_categories.items(), key=operator.itemgetter(1), reverse=True):

                # sets the tag at the specified index as the most likely tag
                tags_pred[counter] = k

                # ends the loop after the first tag is taken
                break

            # adds 1 to the counter
            counter = counter + 1

        else:

            # once it has reached the end of the list, the loop breaks
            break

    # imports the confusion matrix from scikit learn

    # inputs the true and predicted tags into the matrix and sets it to variable cm
    cm = confusion_matrix(tags_true, tags_pred, labels=['TRESSPASS', 'FRAUD', 'WEAPON', 'TRAFFIC', 'ASSAULT', 'OBSTRUCTION', 'FINANCIAL'])

    print("Tags, left to right, top to bottom")

    # prints the list of tags in the json file
    print(d['classes'])

    # prints the confusion matrix in the run window
    print(cm)

    # sets disp to ConfusionMatrixDisplay, using cm as the confusion matrix to base it on and the tags as the labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=d['classes'])

    # plots disp
    disp.plot()

    # opens the matrix visualization in a new window
    plt.show()

def main():
    funs()


if __name__ == "__main__":
    main()
