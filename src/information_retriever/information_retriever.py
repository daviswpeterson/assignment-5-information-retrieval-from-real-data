"""Abstract data type definitions for a basic classifier."""

from __future__ import annotations
from abc import ABC, abstractmethod
from classifier_models import *


__author__ = "Mike Ryu"
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Mike Ryu"]
__license__ = "MIT"
__email__ = "mryu@westmont.edu"


class IrFeature(Feature):
    """Feature used classification of an object.

    Attributes:
        _name (str): human-readable name of the feature (e.g., "over 65 years old")
        _value (str): machine-readable value of the feature (e.g., True)
    """

    def __init__(self, name, value=None, value_class=None):
        super().__init__(name, value, value)


class IrFeatureSet(FeatureSet):
    """A set of features that represent a single object. Optionally includes the known class of the object.

    Attributes:
        _feat (set[Feature]): a set of features that define this object for the purposes of a classifier
        _clas (str | None): optional attribute set as the pre-defined classification of this object
    """

    def __init__(self, features: set[IrFeature], known_clas=None):
        super().__init__(features, known_clas)

    @classmethod
    def ir_build(cls, source_object: Any, known_clas=None) -> IrFeatureSet:
        """Method that builds and returns an instance of FeatureSet given a source object that requires preprocessing.

        For instance, a subclass of `FeatureSet` may be designed to take in a text file object as the `source_object`
        build features based on the tokens that are present in the text file. In this subclass, the logic for
        tokenization and instantiation of `Feature` objects based on the tokens should be written in this method.

        The `return` statement in the actual implementation of this method should simply be a call to the
        constructor where `features` argument is the set of `Feature` instances created within the implementation of
        this method.

        :param source_object: object to build the feature set from
        :param known_clas: pre-defined classification of the source object
        :param user_config: user inputted arguments that decide what type of features to focus on
            -Title keywords
            -Channel names
            -Month
            -Time
        :return: an instance of `FeatureSet` built based on the `source_object` passed in

        {
        "header": "YouTube",
        "title": "Watched Ranking Your CURSED Presentations",
        "titleUrl": "https://www.youtube.com/watch?v\u003dcTSvBW3qC6g",
        "subtitles": [{
            "name": "jschlattLIVE",
            "url": "https://www.youtube.com/channel/UC2mP7il3YV7TxM_3m6U0bwA"
        }],
        "time": "2023-08-07T05:27:57.394Z",
        "products": ["YouTube"],
        "activityControls": ["YouTube watch history"]
        }

        """
        return_set = set()
        title_set = set(source_object["title"].split())
        for word in title_set:
            return_set.add(IrFeature("Title contains " + word, True, "title"))
        return_set.add(IrFeature("Channel is " + source_object["subtitle"][0]["name"], True, "channel"))
        return_set.add(IrFeature("Month is " + source_object["time"][6:7], True, "month"))
        return_set.add(IrFeature("Time is after 3pm", int(type(source_object["time"][10:11])) >= 15, "time"))

        return IrFeatureSet(return_set, known_clas)



class IrClassifier(ABC):
    """Abstract definition for an object classifier."""
    def __init__(self, probability_dict: dict):
        # Will have a set of all the features for the twitter_samples and how they predict which class (probability)
        # SO,this constructor should have a dictionary of features and their probability for + or - tweet
        self.probability_dict = probability_dict

    def get_probability_dict(self) -> dict:
        return self.probability_dict

    @abstractmethod
    def ir_gamma(self, a_feature_set: IrFeatureSet) -> str:
        """Given a single feature set representing an object to be classified, returns the most probable class
        for the object based on the training this classifier received (via a call to `train` class method).

        :param a_feature_set: a single feature set representing an object to be classified
        :return: name of the class with the highest probability for the object
        """

        gammaDict = {
            "2017": 1 / 2,
            "2018": 1 / 2,
            "2019": 1 / 2,
            "2020": 1 / 2,
            "2021": 1 / 2,
            "2022": 1 / 2,
            "2023": 1 / 2,
        }

        for feature in a_feature_set.feat:
            if self.probability_dict.get(feature, 0) != 0:  # if the feature is in the dictionary
                gammaDict["2017"] *= self.probability_dict[feature][0]  # further compute gamma for positive
                gammaDict["2018"] *= self.probability_dict[feature][1]  # further compute gamma for positive
                gammaDict["2019"] *= self.probability_dict[feature][2]  # further compute gamma for positive
                gammaDict["2020"] *= self.probability_dict[feature][3]  # further compute gamma for positive
                gammaDict["2021"] *= self.probability_dict[feature][4]  # further compute gamma for positive
                gammaDict["2022"] *= self.probability_dict[feature][5]  # further compute gamma for positive
                gammaDict["2023"] *= self.probability_dict[feature][6]  # further compute gamma for positive

        mostLikelyYear = max(gammaDict, key=gammaDict.get)
        return mostLikelyYear + ", " + str(gammaDict[mostLikelyYear])

    @abstractmethod
    def ir_present_features(self, top_n: int = 1) -> None:
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        pass

    @classmethod
    @abstractmethod
    def ir_train(cls, training_set: Iterable[IrFeatureSet]) -> IrClassifier:
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
        the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
        `present_features` method calls immediately without needing any other method invocations prior to them.

        :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
        :return: an instance of `AbstractClassifier` with its training already completed
        """

        all_features = {}  # {"name": [0, 0, 0, 0, 0, 0, 0]}

        all_classes = {
            2017: [0, 0],  # First value is IrFeatureSet.clas
            2018: [1, 0],
            2019: [2, 0],
            2020: [3, 0],
            2021: [4, 0],
            2022: [5, 0],
            2023: [6, 0]
        }

        for feature_set in training_set:
            for feature in feature_set.feat:

                if all_features.get(feature, 0) == 0:  # if this feature is already recorded in the dict -> +1 to correct class
                    all_features[feature] = [0, 0, 0, 0, 0, 0, 0]  # adds this feature to the dict

                all_features[feature][all_classes[feature_set.clas][0]] += 1

            # add one to pos or negative total depending on which class the feature_set was
            all_classes[feature_set.clas][1] += 1

        # divide each # of positive or negative tweets with a specific feature by the total number of positive or
        # negative tweets respectively
        for feature in all_features.keys():
            all_features[feature][0] /= all_classes[2017][0]
            all_features[feature][1] /= all_classes[2017][1]
            all_features[feature][2] /= all_classes[2017][2]
            all_features[feature][3] /= all_classes[2017][3]
            all_features[feature][4] /= all_classes[2017][4]
            all_features[feature][5] /= all_classes[2017][5]
            all_features[feature][6] /= all_classes[2017][6]

        return IrClassifier(all_features)