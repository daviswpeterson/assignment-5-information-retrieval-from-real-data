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

    # access json
    with importlib.resources.open_text("data", "davis-watch-history.json") as file:
        data = json.load(file)

    # build all the feature sets
    all_videos = []
    for video in data:
        if not video.keys().contains("details"):  # This should filter out ads
            all_videos.append(IrFeatureSet.ir_build(video, video["time"][:4]))

    random.shuffle(all_videos)  # shuffle them

    train_ir_feature_sets = all_videos[:38600]  # 80% for training (8k tweet feature sets)
    test_ir_feature_sets = all_videos[38600:]  # 20% for testing (2k tweet feature sets)

    our_tweet_classifier = IrClassifier.train(train_ir_feature_sets)  # create our classifier

    i = 0
    while i < 10:  # change this to however many we want to see
        print("Actual class: " + test_ir_feature_sets[i].clas + " | Predicted class: "
              + our_tweet_classifier.gamma(test_ir_feature_sets[i]))
        i += 1

    print("\nAccuracy = " + str(accuracy(test_ir_feature_sets, 1000, our_tweet_classifier)) + "\n")

    # our_tweet_classifier.present_features(20)  # present top features used by classifier (change num based on how many)


def accuracy(list_of_sets: list[FeatureSet], amount: int, classifier: AbstractClassifier) -> float:
    i = 0
    accuracyTally = 0
    while i < amount:  # change this to however many we want to see
        if list_of_sets[i].clas in classifier.gamma(list_of_sets[i]):
            accuracyTally += 1
        i += 1
    return accuracyTally / amount

if __name__ == '__main__':
    main()