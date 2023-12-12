
from __future__ import annotations
from __future__ import annotations
from typing import Any, Iterable

__author__ = "Davis Peterson"
__copyright__ = "Copyright 2023, Westmont College, Davis Peterson"
__credits__ = ["Davis Peterson, Connor Rogstad, Mike Ryu"]
__license__ = "MIT"
__email__ = "davpeterson@westmont.edu"


class IrFeature:
    """Feature used classification of an object.

    Attributes:
        _name (str): human-readable name of the feature (e.g., "over 65 years old")
        _value (str): machine-readable value of the feature (e.g., True)
    """

    def __init__(self, name, value=None):
        self._name: str = name
        self._value: Any = value
        self._value_class: Any = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Any:
        return self._value

    @property
    def value_class(self) -> Any:
        return self._value_class

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, IrFeature):
            return False
        else:
            return self._name == other.name and self._value == other.value and self._value_class == other.value_class

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self._name} = {self._value}"

    def __hash__(self) -> int:
        return hash((self._name, self._value))


class IrFeatureSet:
    """A set of features that represent a single object. Optionally includes the known class of the object.

    Attributes:
        _feat (set[Feature]): a set of features that define this object for the purposes of a classifier
        _clas (str | None): optional attribute set as the pre-defined classification of this object
    """

    def __init__(self, features: set[IrFeature], known_clas=None):
        self._feat: set[IrFeature] = features
        self._clas: str | None = known_clas

    @property
    def feat(self):
        return self._feat

    @property
    def clas(self):
        return self._clas

    @classmethod
    def ir_build(cls, source_object: Any, known_clas=None) -> IrFeatureSet:
        """Method that builds and returns an instance of FeatureSet given a source object that requires preprocessing.

        :param source_object: object to build the feature set from
        :param known_clas: pre-defined classification of the source object
        :return: an instance of `FeatureSet` built based on the `source_object` passed in
        """
        return_set = set()
        title_set = set(source_object["title"].lower().split())
        for word in title_set:
            return_set.add(IrFeature("Title contains " + word, True))  # Title keywords
        if source_object.get("subtitles", 0) != 0:
            return_set.add(IrFeature("Channel is " + source_object["subtitles"][0]["name"], True))  # Channel name
        # print(source_object["time"][5:7])
        return_set.add(IrFeature("Month is " + source_object["time"][5:7], True))  # Month
        # print(source_object["time"][11:13])
        return_set.add(IrFeature("Time is after 3pm", int(source_object["time"][11:13]) >= 15))  # Hour

        return IrFeatureSet(return_set, known_clas)


class IrClassifier:
    """Definition for an object classifier."""
    def __init__(self, probability_dict: dict, proportion_dict: dict):
        """
        A fully trained classifier which is able to predict based of a set of classes
        :param probability_dict: a dictionary of every feature in the corpus with a probability for every class for each
        feature.
        :param proportion_dict: the amount of each class that appears in the training set
        """
        self.probability_dict = probability_dict
        self.proportion_dict = proportion_dict

    def get_probability_dict(self) -> dict:
        return self.probability_dict

    def get_proportion_dict(self) -> dict:
        return self.proportion_dict

    def ir_gamma(self, a_feature_set: IrFeatureSet) -> str:
        """Given a single feature set representing an object to be classified, returns the most probable class
        for the object based on the training this classifier received (via a call to `train` class method).

        :param a_feature_set: a single feature set representing an object to be classified
        :return: name of the class with the highest probability for the object
        """

        gammaDict = self.proportion_dict.copy()
        total = 0
        for key in gammaDict.keys():
            total += gammaDict[key]

        for key in gammaDict.keys():
            gammaDict[key] /= total

        for feature in a_feature_set.feat:
            if self.probability_dict.get(feature, 0) != 0:  # if the feature is in the dictionary
                gammaDict["2017"] *= self.probability_dict[feature][0]  # perform the  next gamma calculation on it
                gammaDict["2018"] *= self.probability_dict[feature][1]
                gammaDict["2019"] *= self.probability_dict[feature][2]
                gammaDict["2020"] *= self.probability_dict[feature][3]
                gammaDict["2021"] *= self.probability_dict[feature][4]
                gammaDict["2022"] *= self.probability_dict[feature][5]
                gammaDict["2023"] *= self.probability_dict[feature][6]

        mostLikelyYear = max(gammaDict, key=gammaDict.get)
        return mostLikelyYear + ", " + str(gammaDict[mostLikelyYear])  # year, gamma

    @classmethod
    def ir_train(cls, training_set: Iterable[IrFeatureSet]) -> IrClassifier:
        """Method that builds a classifier instance with its training (supervised learning) already completed.

        :param training_set: An iterable collection of `IrFeatureSet` to use for training the classifier
        :return: an instance of `IrClassifier` with its training already completed
        """

        all_features = {}  # {"name": [0, 0, 0, 0, 0, 0, 0]}
        all_classes = {
            "2017": [0, 0],  # First value is IrFeatureSet.clas, second is its tally
            "2018": [1, 0],
            "2019": [2, 0],
            "2020": [3, 0],
            "2021": [4, 0],
            "2022": [5, 0],
            "2023": [6, 0]
        }

        for feature_set in training_set:
            for feature in feature_set.feat:

                if all_features.get(feature, 0) == 0:  # check if feature is in the dict
                    all_features[feature] = [0, 0, 0, 0, 0, 0, 0]  # adds this feature to the dict if not

                all_features[feature][all_classes[feature_set.clas][0]] += 1  # add a tally to feature probability

            all_classes[feature_set.clas][1] += 1  # add a tally to the amount of the class

        for feature in all_features.keys():  # Finalize the probabilities
            if all_classes["2017"][1] != 0:  # To allow for n classes, divide by 0 errors have to covered
                all_features[feature][0] /= all_classes["2017"][1]  # then further calculate
            if all_classes["2018"][1] != 0:
                all_features[feature][1] /= all_classes["2018"][1]
            if all_classes["2019"][1] != 0:
                all_features[feature][2] /= all_classes["2019"][1]
            if all_classes["2020"][1] != 0:
                all_features[feature][3] /= all_classes["2020"][1]
            if all_classes["2021"][1] != 0:
                all_features[feature][4] /= all_classes["2021"][1]
            if all_classes["2022"][1] != 0:
                all_features[feature][5] /= all_classes["2022"][1]
            if all_classes["2023"][1] != 0:
                all_features[feature][6] /= all_classes["2023"][1]

        proportion_dict = {}
        for key in all_classes.keys():
            proportion_dict[key] = all_classes[key][1]  # create the proportion dictionary

        return IrClassifier(all_features, proportion_dict)
