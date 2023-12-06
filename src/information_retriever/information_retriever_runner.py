"""Query driver for the vector space model using NLTK's Inaugural corpus.
"""

import sys
import json
import random
from information_retriever import *
import importlib.resources

__author__ = "Connor Rogstad and Davis Peterson"
__copyright__ = "Copyright 2023, Westmont College, Connor Rogstad and Davis Peterson"
__credits__ = ["Connor Rogstad", "Davis Peterson"]
__license__ = "MIT"
__email__ = ["crogstad@westmont.edu", "davpeterson@westmont.edu"]


def main() -> None:

    exit_code = False
    retrain = True

    while not exit_code:

        while retrain:

            print("\nTraining a new classifier...")

            # access json
            with importlib.resources.open_text("data", "davis-watch-history.json") as file:
                data = json.load(file)

            # build all the feature sets
            all_videos = []
            for video in data:
                if "details" not in video.keys():  # This should filter out ads
                    all_videos.append(IrFeatureSet.ir_build(video, video["time"][:4]))

            random.shuffle(all_videos)  # shuffle them

            train_ir_feature_sets = all_videos[:30600]  # 80% for training (8k tweet feature sets)
            test_ir_feature_sets = all_videos[30600:]  # 20% for testing (2k tweet feature sets)

            ir_classifier = IrClassifier.ir_train(train_ir_feature_sets)  # create our classifier

            # i = 0
            # while i < 10:  # change this to however many we want to see
            #     # print(str(test_ir_feature_sets[i].feat))
            #     print("Actual class: " + test_ir_feature_sets[i].clas + " | Predicted class: "
            #           + ir_classifier.ir_gamma(test_ir_feature_sets[i]))
            #     i += 1

            print("\nThe accuracy of your current classifier is " + str(accuracy(test_ir_feature_sets, 2000, ir_classifier)))
            yes_or_no = input("\nWould you like to train a new classifier instead? (y/n) ")

            if yes_or_no == "n":
                retrain = False

        print("For the following feature prompts, simply press \"enter\" if you do not wish to test that feature type.")
        titleStr = input("\nEnter a YouTube video title: ")
        channelStr = input("\nEnter a YouTube channel name: ")
        hourStr = input("\nEnter an hour of the day as a two digit number (i.e. 01, 17): ")
        monthStr = input("\nEnter a month of the year as a two digit number (i.e. 01, 11: ")


        # our_tweet_classifier.present_features(20)  # present top features used by classifier (change num based on how many)


def accuracy(list_of_sets: list[IrFeatureSet], amount: int, classifier: IrClassifier) -> float:
    i = 0
    accuracyTally = 0
    while i < amount:  # change this to however many we want to see
        if list_of_sets[i].clas in classifier.ir_gamma(list_of_sets[i]):
            accuracyTally += 1
        i += 1
    return accuracyTally / amount


if __name__ == '__main__':
    main()