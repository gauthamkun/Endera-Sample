# import statements
import spacy
import operator
import json
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
import pandas as pd
# ---------------------------------------- Predicting the Tags ---------------------------------------------------------
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, classification_report, precision_recall_fscore_support


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

    tags_score = [annotations[1]['entities'][0][2] for annotations in d['annotations']]

    print(tags_score)
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
                tags_score[counter] = v
                # ends the loop after the first tag is taken
                break

            # adds 1 to the counter
            counter = counter + 1

            # for k, v in sorted(article_categories.items(), key=operator.itemgetter(1), reverse=True):
            # print(k, ":", v)

        else:

            # once it has reached the end of the list, the loop breaks
            break

    # ---------------------------------------- Confusion Matrix --------------------------------------------------------

    # imports the confusion matrix from scikit learn

    # inputs the true and predicted tags into the matrix and sets it to variable cm
    cm = sklearn.metrics.confusion_matrix(tags_true, tags_pred,
                                          labels=d['classes'])

    print("Tags, left to right, top to bottom")

    # prints the list of tags in the json file
    print(d['classes'])

    # prints the confusion matrix in the run window
    print(cm)

    # sets disp to ConfusionMatrixDisplay, using cm as the confusion matrix to base it on and the tags as the labels
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=d['classes'])

    # plots disp
    disp.plot()

    # opens the matrix visualization in a new window
    plt.show()

    # ---------------------------------------- Recall and Precision ----------------------------------------------------

    # sets the tag labels to a variable
    # (purely just to keep code neat, could replace "tagLabels" with list of tags if needed
    tag_labels = d['classes']

    # prints the precision score of each tag
    # from scikit learn:
    # "The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false
    # positives. The precision is intuitively the ability of the classifier not to label as positive a sample
    # that is negative."
    p_score = sklearn.metrics.precision_score(tags_true, tags_pred, labels=tag_labels, pos_label=1, average=None,
                                              sample_weight=None,
                                              zero_division='warn')

    print("Precision Scores")
    p_counter = 0
    while p_counter < len(p_score):
        print(d['classes'][p_counter], " = ", p_score[p_counter])
        p_counter = p_counter + 1

    print("\n")

    # prints the recall score of each tag
    # from scikit learn:
    # "The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of
    # false negatives. The recall is intuitively the ability of the classifier to find all the positive samples."
    r_score = sklearn.metrics.recall_score(tags_true, tags_pred, labels=tag_labels, pos_label=1, average=None,
                                           sample_weight=None,
                                           zero_division='warn')

    print("Recall Scores")
    r_counter = 0
    while r_counter < len(r_score):
        print(d['classes'][r_counter], " = ", r_score[r_counter])
        r_counter = r_counter + 1

    # Above methods will throw a warning if the dataset has no samples
    # for example, if a tag has 0 true tag cases, it will attempt to divide by 0

    # --------------------------------------- Classification Report ----------------------------------------------------
    class_report = classification_report(tags_true, tags_pred, labels=tag_labels, target_names=tag_labels,
                                         output_dict=True)
    sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True)
    plt.show()

    # --------------------------------------- Precision Recall Curve ---------------------------------------------------
    var2 = precision_recall_fscore_support(tags_true, tags_pred, labels=tag_labels)

    print("\n")
    print(var2[3])
    print("\n")

    # make a bar graph for f1 scores
    # get one hot encoded data from gautham
    # and score values

    # precision = dict()
    # recall = dict()
    # for i in range(len(tag_labels)):
    #    precision[i], recall[i], _ = precision_recall_curve(tags_true[:, i],
    #                                                        tags_score[:, i])
    #    plt.plot(recall[i], precision[i], lw=2, label=tag_labels.format(i))

    # plt.xlabel("recall")
    # plt.ylabel("precision")
    # plt.legend(loc="best")
    # plt.title("precision vs. recall curve")
    # plt.show()


def main():
    funs()


if __name__ == "__main__":
    main()
